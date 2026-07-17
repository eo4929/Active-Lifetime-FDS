import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Confidence:
    def __init__(self, enriched_features, enriched_features_df, transaction_features, account_features, customer_features):
        enriched_features_df = pd.DataFrame(
            enriched_features,
            columns=[f"enriched_feature_{i + 1}" for i in range(enriched_features.shape[1])]
        )

        enriched_features_df['label'] = df['transaction_label'].values  # 레이블 추가
        enriched_features_df['active_lifetime'] = df['active_lifetime'].values  # 활성 구간 추가

        self.enriched_features_df = enriched_features_df
        self.transaction_features = transaction_features
        self.account_features = account_features
        self.customer_features = customer_features

    def calculate_similarity(self, x_i, fraud_set, level_features, original = 'True'):
        similarities = []

        for fraud_transaction in fraud_set:
            if original == 'True':
                original_similarity = cosine_similarity(x_i['original_features'].reshape(1, -1), fraud_transaction['original_features'].reshape(1, -1))
                level_similarity = cosine_similarity(
                    level_features[x_i['index']].reshape(1, -1), level_features[fraud_transaction['index']].reshape(1, -1)
                )
                similarities.append(original_similarity.sum() + level_similarity.sum())

        return np.sum(similarities)

    def calculate_confidence(self, x_i, fraud_set, enriched_features, transaction_features, account_features, customer_features):
        numerator = self.calculate_similarity(x_i, fraud_set, transaction_features)

        denominator = sum(
            calculate_similarity(normal_transaction, fraud_set, transaction_features)
            + calculate_similarity(normal_transaction, fraud_set, account_features)
            + calculate_similarity(normal_transaction, fraud_set, customer_features)
            for normal_transaction in enriched_features
        )

        return 1 - (np.log(1 + numerator) / np.log(1 + max(denominator, 1e-10)))

    def sample_high_confidence(self, enriched_features, transaction_features, account_features, customer_features, k=2):
        sampled_indices = []

        for al in enriched_features['active_lifetime'].unique():
            al_df = enriched_features[enriched_features['active_lifetime'] == al]
            fraud_transactions = al_df[al_df['label'] == 1]
            normal_transactions = al_df[al_df['label'] == 0]

            if len(fraud_transactions) == 0 or len(normal_transactions) == 0:
                continue

            fraud_set = fraud_transactions.to_dict('records')
            normal_set = normal_transactions.to_dict('records')

            confidences = []
            for normal_transaction in normal_set:
                confidence = calculate_confidence(
                    normal_transaction, fraud_set, normal_set,
                    transaction_features, account_features, customer_features
                )
                confidences.append((normal_transaction['index'], confidence))

            confidences = sorted(confidences, key=lambda x: x[1], reverse=True)
            num_to_sample = k * len(fraud_transactions)
            sampled_indices.extend([idx for idx, _ in confidences[:num_to_sample]])

        return enriched_features.loc[sampled_indices]

    def create_enriched_features_with_positive_samples(self, enriched_features_df, transaction_features, account_features, customer_features, k=2):

        sampled_df = sample_high_confidence(
            enriched_features_df, transaction_features, account_features, customer_features, k
        )

        fraud_set_df = enriched_features_df[(enriched_features_df['label'] == 1) & (enriched_features_df['active_lifetime'] > 0)]

        non_active_lifetime_df = enriched_features_df[enriched_features_df['active_lifetime'] == 0]

        enriched_features_df_with_positive_samples = pd.concat(
            [sampled_df, fraud_set_df, non_active_lifetime_df], ignore_index=True
        )

        return enriched_features_df_with_positive_samples

    def final_enriched_features_df_with_positive_samples(self):
        enriched_features_df_with_positive_samples = create_enriched_features_with_positive_samples(self.enriched_features_df, self.transaction_features, self.account_features, self.customer_features)
