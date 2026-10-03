from dataclasses import dataclass

import numpy as np
import shap
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from .models import xgboost_classifier
from .schema import LEVELS
from .signed_graph import K_RATIOS, MINKOWSKI_P, SignedKNNGraph, select_k_ratios


def torch_adjacency(adjacency, device):
    """Preserve signed sparse weights on CPU or CUDA, with native autograd."""
    coo = adjacency.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device, check_invariants=True).coalesce()


class SAGEConv(nn.Module):
    """GraphSAGE layer with the mean aggregator: W_self h_v + W_neigh mean_{u in N(v)} s_uv w_uv h_u."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x_dst, x_src, adjacency):
        return self.self_linear(x_dst) + self.neighbor_linear(torch.sparse.mm(adjacency, x_src))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward(self, x, adjacency, x_query=None, query_adjacency=None):
        h = torch.relu(self.conv1(x, x, adjacency))
        if x_query is None:
            return self.conv2(h, h, adjacency)
        h_query = torch.relu(self.conv1(x_query, x, query_adjacency))
        return self.conv2(h_query, h, query_adjacency)


@dataclass
class EnrichedFeatures:
    original: np.ndarray
    transaction: np.ndarray
    account: np.ndarray
    customer: np.ndarray

    @property
    def final(self):
        return np.hstack([self.original, self.transaction, self.account, self.customer])

    def views(self):
        blocks = {
            "original": self.original,
            "transaction": self.transaction,
            "account": self.account,
            "customer": self.customer,
        }
        return {name: MinMaxScaler().fit_transform(block) if len(block) else block for name, block in blocks.items()}


def normalized_shap_importance(X, y, rows, seed=42):
    """phi_i: min-max normalized mean |SHAP| of each original feature over the given rows.

    An XGBoost classifier is trained on all training transactions (original features), and exact
    TreeSHAP values of its log-odds output are computed for every transaction in `rows`.
    """
    if len(X[rows]) == 0 or len(np.unique(y)) < 2:
        return np.zeros(X.shape[1])
    model = xgboost_classifier(seed).fit(X, y)
    phi = np.abs(shap.TreeExplainer(model).shap_values(X[rows])).mean(axis=0)
    span = phi.max() - phi.min()
    return (phi - phi.min()) / span if span > 0 else np.zeros_like(phi)


class ActiveLifetimeFeatureEnricher:
    """Idea 1: active lifetime-driven feature enrichment.

    a_L = GNN(G_L) * P_L(a_L * W_L) for L in {transaction, account, customer},
    w_i = phi_i * (alpha_T * Avg(I_fraud) + (1 - alpha_T) * Avg(F_fraud)),
    a_final = a_original + a_trans + a_account + a_customer (concatenation).

    The positive and negative K ratios of each level's signed K-NN graph are selected by graph
    topology and connectivity analysis (a local heuristic, not the cited method's verified code).
    Held-out transactions use fold-local supervised models and reference graphs.
    P_L is a fixed dimension bridge; extraction_dim=None uses the body's unprojected formula.
    """

    def __init__(
        self,
        alpha_t=0.3,
        hidden_dim=32,
        epochs=50,
        lr=0.01,
        n_folds=5,
        k_ratios=K_RATIOS,
        topology_sample_size=500,
        graph_sample_ratio=0.2,
        graph_max_nodes=None,
        seed=42,
        extraction_dim=32,
        query_batch_size=1024,
        device="cpu",
        k_overrides=None,
        neighbor_backend="approximate",
        approximation_eps=0.1,
        minkowski_p=MINKOWSKI_P,
    ):
        self.alpha_t = alpha_t
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.n_folds = n_folds
        self.k_ratios = k_ratios
        self.topology_sample_size = topology_sample_size
        if not 0 < graph_sample_ratio <= 1:
            raise ValueError("graph_sample_ratio must be in (0, 1]")
        self.graph_sample_ratio = graph_sample_ratio
        self.graph_max_nodes = graph_max_nodes
        self.seed = seed
        if extraction_dim is not None and extraction_dim < 1:
            raise ValueError("extraction_dim must be positive or None")
        if query_batch_size < 1 or n_folds < 2 or epochs < 1:
            raise ValueError("query_batch_size and epochs must be positive; n_folds must be >= 2")
        self.extraction_dim = extraction_dim
        self.query_batch_size = query_batch_size
        self.device = torch.device(device)
        self.k_overrides = k_overrides or {}
        self.neighbor_backend = neighbor_backend
        self.approximation_eps = approximation_eps
        self.minkowski_p = minkowski_p

    def _parameters(self):
        return dict(alpha_t=self.alpha_t, hidden_dim=self.hidden_dim, epochs=self.epochs,
                    lr=self.lr, n_folds=self.n_folds, k_ratios=self.k_ratios,
                    topology_sample_size=self.topology_sample_size,
                    graph_sample_ratio=self.graph_sample_ratio, graph_max_nodes=self.graph_max_nodes,
                    seed=self.seed, extraction_dim=self.extraction_dim,
                    query_batch_size=self.query_batch_size, device=str(self.device), k_overrides=self.k_overrides,
                    neighbor_backend=self.neighbor_backend, approximation_eps=self.approximation_eps,
                    minkowski_p=self.minkowski_p)

    def _projection(self, in_dim):
        out_dim = self.extraction_dim or in_dim
        # Fixed feature grouping, independent of labels and folds. No dropped
        # coordinates; identity when out_dim == in_dim. This fills the paper's
        # missing dimension bridge between elementwise fusion and 32 features.
        projection = np.zeros((in_dim, out_dim), dtype=np.float32)
        groups = np.arange(in_dim) % out_dim
        projection[np.arange(in_dim), groups] = 1.0 / np.bincount(groups, minlength=out_dim)[groups]
        return projection

    def fit(self, levels, y, in_active_lifetime, avg_intensity, avg_frequency):
        y = np.asarray(y)
        original = np.hstack([levels[level] for level in LEVELS])
        self.shap_sample_size_ = int(np.sum(in_active_lifetime))
        self.phi_ = phi = normalized_shap_importance(original, y, in_active_lifetime, self.seed)
        weights = phi * (self.alpha_t * avg_intensity + (1 - self.alpha_t) * avg_frequency)
        boundaries = np.cumsum([levels[level].shape[1] for level in LEVELS])[:-1]
        self.weights_ = dict(zip(LEVELS, np.split(weights, boundaries)))
        self.projections_ = {level: self._projection(levels[level].shape[1]) for level in LEVELS}
        self.enabled_ = bool(self.shap_sample_size_ and np.any(weights))

        self.topology_, self.graphs_, self.gnns_ = {}, {}, {}
        if not self.enabled_:
            return self
        for level in LEVELS:
            if level in self.k_overrides:
                positive_ratio, negative_ratio = self.k_overrides[level]
                self.topology_[level] = None
            else:
                positive_ratio, negative_ratio, self.topology_[level] = select_k_ratios(
                    levels[level], y, self.k_ratios, self.topology_sample_size, p=self.minkowski_p, seed=self.seed
                )
            graph = self._graph(positive_ratio, negative_ratio, len(y)).fit(levels[level], y)
            self.graphs_[level], self.gnns_[level] = graph, self._train_gnn(graph)
        return self

    def transform(self, levels):
        if not self.enabled_:
            return self._zero_features(levels)
        embeddings = {level: self._embed(self.graphs_[level], self.gnns_[level], levels[level]) for level in LEVELS}
        return self._fuse(levels, embeddings)

    def fit_transform(self, levels, y, in_active_lifetime, avg_intensity, avg_frequency, fold_context=None):
        y = np.asarray(y)
        if len(y) < 2 or not np.any(in_active_lifetime) or len(np.unique(y)) < 2:
            self.fit(levels, y, in_active_lifetime, avg_intensity, avg_frequency)
            self.oof_splits_ = []
            return self._zero_features(levels)
        counts = np.unique(y, return_counts=True)[1]
        n_splits = min(self.n_folds, int(counts.min()))
        folds = (StratifiedKFold(n_splits, shuffle=True, random_state=self.seed) if n_splits >= 2
                 else KFold(min(self.n_folds, len(y)), shuffle=True, random_state=self.seed))
        self.oof_splits_ = [(a.copy(), b.copy()) for a, b in folds.split(levels[LEVELS[0]], y)]
        fused = {level: np.zeros((len(y), self.extraction_dim or levels[level].shape[1]), dtype=np.float32) for level in LEVELS}
        for fit_rows, held_out in self.oof_splits_:
            child = type(self)(**self._parameters())
            context = (fold_context(fit_rows) if fold_context is not None else
                       (np.asarray(in_active_lifetime)[fit_rows], avg_intensity, avg_frequency))
            child.fit({level: levels[level][fit_rows] for level in LEVELS}, y[fit_rows], *context)
            transformed = child.transform({level: levels[level][held_out] for level in LEVELS})
            for level in LEVELS:
                fused[level][held_out] = getattr(transformed, level)
        # Fit the deployment model only AFTER producing held-out features. Its
        # parameters are never used for training-row OOF representations.
        self.fit(levels, y, in_active_lifetime, avg_intensity, avg_frequency)
        return EnrichedFeatures(original=np.hstack([levels[level] for level in LEVELS]), **fused)

    def _zero_features(self, levels):
        return EnrichedFeatures(original=np.hstack([levels[level] for level in LEVELS]),
                                **{level: np.zeros((len(levels[level]), self.extraction_dim or levels[level].shape[1])) for level in LEVELS})

    def _graph(self, positive_ratio, negative_ratio, n_rows):
        """Reference graph over a stratified sample of graph_sample_ratio of the rows, capped by graph_max_nodes."""
        nodes = max(2, int(np.ceil(self.graph_sample_ratio * n_rows)))
        if self.graph_max_nodes is not None:
            nodes = min(nodes, self.graph_max_nodes)
        return SignedKNNGraph(positive_ratio, negative_ratio, nodes, p=self.minkowski_p, seed=self.seed,
                              neighbor_backend=self.neighbor_backend, approximation_eps=self.approximation_eps)

    def _fuse(self, levels, embeddings):
        fused = {level: embeddings[level] * ((levels[level] * self.weights_[level]) @ self.projections_[level]) for level in LEVELS}
        return EnrichedFeatures(original=np.hstack([levels[level] for level in LEVELS]), **fused)

    def _train_gnn(self, graph):
        """Supervised node classification (normal vs. fraud) with class-weighted cross-entropy.

        The output of the second SAGEConv layer, which feeds the classifier head, is GNN(G_L).
        """
        torch.manual_seed(self.seed)
        x = torch.tensor(graph.X_, device=self.device)
        adjacency = torch_adjacency(graph.adjacency_, self.device)
        target = torch.tensor(graph.y_, dtype=torch.long, device=self.device)
        counts = np.bincount(graph.y_, minlength=2)
        class_weight = torch.tensor(len(graph.y_) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=self.device)

        model = GraphSAGE(x.shape[1], self.hidden_dim, self.extraction_dim or x.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = F.cross_entropy(model.classifier(model(x, adjacency)), target, weight=class_weight)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            model.reference_x_ = x
            model.reference_hidden_ = torch.relu(model.conv1(x, x, adjacency))
        return model

    @torch.no_grad()
    def _embed(self, graph, gnn, X):
        output = np.empty((len(X), gnn.conv2.self_linear.out_features), dtype=np.float32)
        for start in range(0, len(X), self.query_batch_size):
            chunk = X[start:start + self.query_batch_size]
            x_query = torch.tensor(chunk, dtype=torch.float32, device=self.device)
            adjacency = torch_adjacency(graph.query_adjacency(chunk), self.device)
            h_query = torch.relu(gnn.conv1(x_query, gnn.reference_x_, adjacency))
            output[start:start + len(chunk)] = gnn.conv2(h_query, gnn.reference_hidden_, adjacency).cpu().numpy()
        return output
