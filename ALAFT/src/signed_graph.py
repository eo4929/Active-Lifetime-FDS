import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

K_RATIOS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GRAPH_MEASUREMENTS = ["diameter", "average_path_length", "reachable_pairs", "reciprocity"]
MINKOWSKI_P = 3


class ApproximateMinkowskiIndex:
    """SciPy's bounded approximate search; kth distance is <= (1+eps) exact."""

    def __init__(self, X, p, eps):
        self.tree = cKDTree(X)
        self.p, self.eps = p, eps

    def kneighbors(self, X, n_neighbors):
        distance, indices = self.tree.query(X, k=n_neighbors, p=self.p, eps=self.eps, workers=-1)
        return np.asarray(distance).reshape(len(X), n_neighbors), np.asarray(indices).reshape(len(X), n_neighbors)


def stratified_sample(y, size, seed=42):
    """Row indices of a stratified sample in which every class keeps at least size / (2 * #classes) rows."""
    y = np.asarray(y)
    if size is None or len(y) <= size:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(np.round(size * counts / counts.sum()), np.minimum(counts, size // (2 * len(classes))))
    quota = quota.astype(int)
    quota[counts.argmax()] -= quota.sum() - size
    rows = [rng.choice(np.flatnonzero(y == c), q, replace=False) for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(rows))


def k_from_ratio(ratio, max_connections):
    """K = ratio x (maximum possible connections), at least 2 as in the candidate range of K."""
    return int(min(max_connections, max(2, round(ratio * max_connections))))


class SignedKNNGraph:
    """Signed directed K-NN graph with similarity 1 / (1 + Minkowski distance of order p).

    A node of class c points to its K+_c nearest nodes of the same class through positive edges
    and to its K-_c nearest nodes of a different class through negative edges. K is given as a
    ratio of the maximum possible connections: K+_c = r+ (|c| - 1) and K-_c = r- (n - |c|).
    With `max_nodes`, the graph is built over a stratified sample of that many nodes.
    """

    def __init__(self, positive_ratio, negative_ratio, max_nodes=None, p=MINKOWSKI_P, seed=42,
                 neighbor_backend="exact", approximation_eps=0.1):
        if not 0 < positive_ratio <= 1 or not 0 < negative_ratio <= 1:
            raise ValueError("K ratios must be in (0, 1]")
        if p < 1:
            raise ValueError("the Minkowski order must be >= 1")
        if neighbor_backend not in {"exact", "approximate"} or approximation_eps < 0:
            raise ValueError("invalid nearest-neighbor backend or approximation bound")
        self.positive_ratio = positive_ratio
        self.negative_ratio = negative_ratio
        self.max_nodes = max_nodes
        self.p = p
        self.seed = seed
        self.neighbor_backend = neighbor_backend
        self.approximation_eps = approximation_eps

    def fit(self, X, y):
        if len(y) == 0:
            raise ValueError("a reference graph requires at least one transaction")
        nodes = stratified_sample(y, self.max_nodes, self.seed)
        self.X_ = np.asarray(X, dtype=np.float32)[nodes]
        self.y_ = np.asarray(y)[nodes]
        self.classes_, counts = np.unique(self.y_, return_counts=True)
        self.k_pos_ = {c: k_from_ratio(self.positive_ratio, n_c - 1) for c, n_c in zip(self.classes_, counts)}
        self.k_neg_ = {c: k_from_ratio(self.negative_ratio, len(self.y_) - n_c) for c, n_c in zip(self.classes_, counts)}

        self._same = {c: self._index(self.y_ == c) for c in self.classes_}
        self._different = {c: self._index(self.y_ != c) for c in self.classes_}
        self._all = self._index(np.ones(len(self.y_), dtype=bool))

        positive = self._edges(self._same, self.X_, self.y_, self.k_pos_, exclude_self=True)
        negative = self._edges(self._different, self.X_, self.y_, self.k_neg_)
        self.adjacency_ = self._signed_adjacency(positive, negative, len(self.y_))
        return self

    def query_adjacency(self, X):
        """Connects unseen nodes to the graph, signing edges against their K-NN majority class."""
        X = np.asarray(X, dtype=np.float32)
        ids, nn = self._all
        if len(X) == 0:
            return csr_matrix((0, len(self.y_)), dtype=np.float32)
        vote_k = min(len(self.y_), max(1, min(self.k_pos_.values())))
        _, positions = nn.kneighbors(X, n_neighbors=vote_k)
        votes = (self.y_[ids[positions]][:, :, None] == self.classes_).sum(axis=1)
        labels = self.classes_[votes.argmax(axis=1)]

        positive = self._edges(self._same, X, labels, self.k_pos_)
        negative = self._edges(self._different, X, labels, self.k_neg_)
        return self._signed_adjacency(positive, negative, len(X))

    def _index(self, mask):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return ids, None
        if self.neighbor_backend == "approximate":
            return ids, ApproximateMinkowskiIndex(self.X_[ids], self.p, self.approximation_eps)
        return ids, NearestNeighbors(metric="minkowski", p=self.p, n_jobs=-1).fit(self.X_[ids])

    def _edges(self, index_by_class, X, labels, k_by_class, exclude_self=False):
        rows, columns, distances = [], [], []
        for c in self.classes_:
            query, k = np.flatnonzero(labels == c), k_by_class[c]
            if len(query) == 0 or k == 0:
                continue
            ids, nn = index_by_class[c]
            distance, positions = nn.kneighbors(X[query], n_neighbors=k + exclude_self)
            neighbors = ids[positions]
            if exclude_self:
                keep = neighbors != query[:, None]
                keep[keep.all(axis=1), -1] = False
                neighbors, distance = neighbors[keep].reshape(-1, k), distance[keep].reshape(-1, k)
            rows.append(np.repeat(query, k))
            columns.append(neighbors.ravel())
            distances.append(distance.ravel())
        if not rows:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
        return np.concatenate(rows), np.concatenate(columns), np.concatenate(distances)

    def _signed_adjacency(self, positive, negative, n_query):
        rows = np.concatenate([positive[0], negative[0]])
        columns = np.concatenate([positive[1], negative[1]])
        signs = np.concatenate([np.ones(len(positive[0])), -np.ones(len(negative[0]))])
        similarity = 1.0 / (1.0 + np.concatenate([positive[2], negative[2]]))
        degree = np.bincount(rows, minlength=n_query)
        values = (signs * similarity / degree[rows]).astype(np.float32)
        return csr_matrix((values, (rows, columns)), shape=(n_query, len(self.y_)))


def graph_measurements(adjacency):
    """Topology and connectivity measurements of a directed graph, regardless of edge signs and weights."""
    structure = (adjacency != 0).astype(np.int8)
    distances = shortest_path(structure, directed=True, unweighted=True)
    np.fill_diagonal(distances, np.inf)
    reachable = distances[np.isfinite(distances)]
    return {
        "diameter": reachable.max() if reachable.size else 0.0,
        "average_path_length": reachable.mean() if reachable.size else 0.0,
        "reachable_pairs": reachable.size,
        "reciprocity": structure.multiply(structure.T).nnz / structure.nnz if structure.nnz else 0.0,
        "strongly_connected_components": connected_components(structure, directed=True, connection="strong")[0],
    }


def select_k_ratios(X, y, ratios=K_RATIOS, sample_size=500, p=MINKOWSKI_P, seed=42):
    """Graph topology and connectivity analysis for selecting the positive and negative K ratios.

    Signed directed K-NN graphs are built for every pair of candidate ratios, and the pair at which
    diameter, average path length, reachable pairs and reciprocity (each min-max normalized over the
    candidates) jointly reach their highest point is selected. The number of strongly connected
    components is reported but not used for the selection.
    """
    rows = stratified_sample(y, sample_size, seed)
    X, y = np.asarray(X)[rows], np.asarray(y)[rows]
    analysis = pd.DataFrame([
        {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            **graph_measurements(SignedKNNGraph(positive_ratio, negative_ratio, p=p).fit(X, y).adjacency_),
        }
        for positive_ratio in ratios
        for negative_ratio in ratios
    ])
    measurements = analysis[GRAPH_MEASUREMENTS]
    span = measurements.max() - measurements.min()
    analysis["score"] = ((measurements - measurements.min()) / span.replace(0, np.nan)).fillna(1.0).mean(axis=1)
    best = analysis.loc[analysis["score"].idxmax()]
    return float(best["positive_ratio"]), float(best["negative_ratio"]), analysis
