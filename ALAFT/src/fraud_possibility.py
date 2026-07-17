import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

class FraudPossibilityComputer:

    def calculate_delta_ij(self, normal_time, fraud_time):
        time_diff = abs(normal_time - fraud_time)
        return 1 / (1 + np.exp(time_diff))

    def calculate_summation(self, normal_trans, fraud_trans, theta, ks):
        numerator = 0
        denominator = 0

        for g in range(-theta, theta + 1):
            avg_fraud_trans = np.mean([fraud_features[j] for j in range(len(fraud_features)) if j + g >= 0], axis=0)

            numerator += np.dot(normal_trans, avg_fraud_trans)

            for fraud_trans in lst_fraud_trans:
              denominator += np.dot(fraud_trans, avg_fraud_trans)
              for k in ks:
                delta = calculate_delta_ij(k, fraud_trans)
                denominator += (delta * denominator)

        return numerator, denominator

    def calculate_fraud_possibility(self, normal_transaction, fraud_transactions, theta):
        fraud_possibility = 0

        for fraud_transaction in fraud_transactions:
            delta_ij = calculate_delta_ij(normal_transaction['timestamp'], fraud_transaction['timestamp'])

            numerator, denominator = calculate_summation(
                normal_transaction['original_features'], fraud_transaction['original_features'], theta
            )

            fraud_possibility += delta_ij * (numerator / max(denominator, 1e-10))

        for fraud_transaction in fraud_transactions:
            delta_ij = calculate_delta_ij(normal_transaction['timestamp'], fraud_transaction['timestamp'])

            numerator, denominator = calculate_summation(
                normal_transaction['transaction_features'], fraud_transaction['transaction_features'], theta
            )

            fraud_possibility += delta_ij * (numerator / max(denominator, 1e-10))

        for fraud_transaction in fraud_transactions:
            delta_ij = calculate_delta_ij(normal_transaction['timestamp'], fraud_transaction['timestamp'])

            numerator, denominator = calculate_summation(
                normal_transaction['account_features'], fraud_transaction['account_features'], theta
            )

            fraud_possibility += delta_ij * (numerator / max(denominator, 1e-10))

        for fraud_transaction in fraud_transactions:
            delta_ij = calculate_delta_ij(normal_transaction['timestamp'], fraud_transaction['timestamp'])

            numerator, denominator = calculate_summation(
                normal_transaction['customer_features'], fraud_transaction['customer_features'], theta
            )

            fraud_possibility += delta_ij * (numerator / max(denominator, 1e-10))

        return fraud_possibility


    def add_fraud_possibility_scores(self, df, active_lifetime_ids, theta=5):
        fraud_df = df[df['label'] == 1]
        normal_df = df[(df['label'] == 0) & (df['active_lifetime'].isin(active_lifetime))]

        fraud_transactions = fraud_df.to_dict('records')
        normal_transactions = normal_df.to_dict('records')

        fraud_possibilities = []
        for normal_transaction in normal_transactions:
            score = calculate_fraud_possibility(
                normal_transaction, fraud_transactions, theta
            )
            fraud_possibilities.append(score)

        normal_df['fraud_possibility'] = fraud_possibilities
        return pd.concat([normal_df, fraud_df], ignore_index=True)


    def weighted_logloss(self, preds, dtrain):
        labels = dtrain.get_label()
        weights = dtrain.get_weight()
        preds = 1 / (1 + np.exp(-preds))

        grad = weights * (preds - labels)
        hess = weights * preds * (1 - preds)

        return grad, hess


    def train_xgboost_with_fraud_possibility(self, df):
        features = df.drop(columns=['label', 'fraud_possibility'])
        target = df['label']
        weights = df['fraud_possibility']

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            features, target, weights, test_size=0.2, random_state=42
        )

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

        eval_results = {}

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            obj=weighted_logloss,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=eval_results,
            verbose_eval=True
        )

        return model, eval_results, pd.DataFrame(dtrain), pd.DataFrame(dtest)