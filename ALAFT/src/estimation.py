import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import mode

class Estimator:
    def __init__(self, train_set_with_fraud_score, test_set_with_fraud_score):
        self.train_set_with_fraud_score = train_set_with_fraud_score
        self.test_set_with_fraud_score = test_set_with_fraud_score

        self.df_with_fraud_score = pd.Concat(train_set_with_fraud_score, test_set_with_fraud_score)

        self.target = 'active_lifetime'

    def calculate_temporal_influence(self, diff_t, rho):
        return np.exp(-rho * diff_t)

    def adjust_fraud_scores(self, df, active_lifetimes, rho=0.1):
        adjusted_scores = []

        for idx, row in df.iterrows():
            transaction_time = row['timestamp']
            fraud_score = row['fraud_score']

            is_active = any(start <= transaction_time <= end for start, end in active_lifetimes)

            if is_active:
                adjusted_scores.append(fraud_score)
            else:
                # Non-active lifetime transactions: adjust fraud score
                diff_t_prev = min(
                    [abs(transaction_time - end) for start, end in active_lifetimes if transaction_time > end],
                    default=np.inf
                )
                diff_t_next = min(
                    [abs(start - transaction_time) for start, end in active_lifetimes if transaction_time < start],
                    default=np.inf
                )

                # Calculate temporal influence factors
                T_I_prev = calculate_temporal_influence(diff_t_prev, rho)

                if diff_t_next == np.inf:
                    # No next active lifetime
                    adjusted_score = fraud_score * T_I_prev
                else:
                    T_I_next = calculate_temporal_influence(diff_t_next, rho)
                    adjusted_score = fraud_score * ((T_I_prev + T_I_next) / 2)

                adjusted_scores.append(adjusted_score)

        df['adjusted_fraud_score'] = adjusted_scores
        return df


    def estimate_active_lifetimes(self, train_df, test_df, target, model_weights=None):
        if model_weights is None:
            model_weights = [1.0, 1.0, 1.0]

        X_train = train_df.drop(columns=[target, 'time_slot', 'timestamp'])
        y_train = train_df[target]
        X_test = test_df.drop(columns=['time_slot', 'timestamp'])

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        tabnet_model = TabNetClassifier()
        tabnet_model.fit(X_train.values, y_train.values, max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128)
        tabnet_preds = tabnet_model.predict(X_test.values)

        predictions = np.array([rf_preds, xgb_preds, tabnet_preds])
        weighted_preds = np.average(predictions, axis=0, weights=model_weights)
        final_preds = (weighted_preds >= 0.5).astype(int)

        test_df['predicted_active_lifetime'] = final_preds

        active_lifetimes = []
        for _, group in test_df[test_df['predicted_active_lifetime'] == 1].groupby('time_slot'):
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            active_lifetimes.append((start_time, end_time))

        return active_lifetimes

    def make_final_transactions_with_score(self, transactions_df):
        estimated_active_lifetimes = self.estimate_active_lifetimes(self.train_set_with_fraud_score, self.test_set_with_fraud_score, self.target)
        self.df_with_adjusted_fraud_score = adjust_fraud_scores(transactions_df, estimated_active_lifetimes, rho=0.1)

        return self.df_with_adjusted_fraud_score