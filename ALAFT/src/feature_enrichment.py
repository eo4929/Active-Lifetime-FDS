from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

from dgl.nn import SAGEConv
import dgl
dgl.backend.backend_name = 'pytorch'


import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors



class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

class GraphConstructor:
    def __init__(self):
        self.transaction_features = df[['transaction_features']].values
        self.account_features = df[['account_features']].values
        self.customer_features = df[['customer_features']].values

        self.transaction_labels = df['transaction_class_label'].values
        self.account_labels = df['account_class_label'].values
        self.customer_labels = df['customer_class_label'].values

        self.embedded_features_t = None
        self.embedded_features_a = None
        self.embedded_features_c = None

    def create_signed_knn_graph(self, features, labels, k=10):
        knn = NearestNeighbors(n_neighbors=k, metric='minkowski').fit(features)
        distances, indices = knn.kneighbors(features)

        G = nx.Graph()

        for i in range(len(features)):
            G.add_node(i, feature=features[i], label=labels[i])

        for i in range(len(features)):
            for j_idx in range(1, k):  # 자신과 연결 제외
                j = indices[i][j_idx]
                distance = distances[i][j_idx]
                weight = 1 / (1 + distance)  # 민코우스키 거리의 역수
                sign = 1 if labels[i] == labels[j] else -1  # 같은 클래스: +1, 다른 클래스: -1
                G.add_edge(i, j, weight=weight, sign=sign)

        return G

    def contruct_multi_level_graphs(self):
        self.G_t = create_signed_knn_graph(self.transaction_features, transaction_labels, k=10)
        self.G_a = create_signed_knn_graph(self.account_features, account_labels, k=10)
        self.G_c = create_signed_knn_graph(self.customer_features, customer_labels, k=10)

        dgl_G_t = dgl.from_networkx(G_t)
        dgl_G_a = dgl.from_networkx(G_a)
        dgl_G_c = dgl.from_networkx(G_c)

        # Prepare feature tensors
        features_t = torch.FloatTensor(transaction_features.values)
        features_a = torch.FloatTensor(account_features.values)
        features_c = torch.FloatTensor(customer_features.values)

        hidden_size = 64  # Hidden layer size (you can tune this)
        output_size = len(df.columns)  # Embedding size (this will be your final GNN feature size)

        # Initialize models
        gnn_model_t = GNNModel(features_t.shape[1], hidden_size, output_size)
        gnn_model_a = GNNModel(features_a.shape[1], hidden_size, output_size)
        gnn_model_c = GNNModel(features_c.shape[1], hidden_size, output_size)

        self.embedded_features_t = gnn_model_t(dgl_G_t, features_t)
        self.embedded_features_a = gnn_model_a(dgl_G_a, features_a)
        self.embedded_features_c = gnn_model_c(dgl_G_c, features_c)

        return self.embedded_features_t, self.embedded_features_a, self.embedded_features_c

class FeatureEnricher:
    def __init__(self):
        shap_values = shap.TreeExplainer(XGBClassifier()).shap_values(df.drop(['transaction_class_label', 'account_class_label','customer_class_label','label_type'], axis=1))
        shap_importance = MinMaxScaler().fit_transform(shap_values)

        # Compute weighted fusion
        I_avg = df.loc[df['active_lifetime'] == 1, 'fraud_intensity'].mean()
        F_avg = df.loc[df['active_lifetime'] == 1, 'fraud_frequency'].mean()
        alpha_T = 0.5  # Adjust as needed

        self.weights = shap_importance * (alpha_T * I_avg + (1 - alpha_T) * F_avg)

        constructor = GraphConstructor()
        self.embedded_features_t, self.embedded_features_a, self.embedded_features_c = constructor.contruct_multi_level_graphs()

    def concatenate_features(self):

        final_transaction_features = self.embedded_features_t * features_t * weights
        final_account_features = self.embedded_features_a * features_a * weights
        final_customer_features = self.embedded_features_c * features_c * weights

        enriched_features= np.concatenate(
            [final_transaction_features.detach().numpy(),
             final_account_features.detach().numpy(),
             final_customer_features.detach().numpy()],
            axis=1
        )

        return enriched_features