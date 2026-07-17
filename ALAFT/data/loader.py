import pandas as pd
import numpy as np

def load_data(raw_data):
  df = raw_data

  I_threshold = 1.2
  F_threshold = 4.0

  df['fraud_intensity'] = df['fraud_amount'] / df['total_amount']
  df['fraud_frequency'] = df['fraud_count'] / df['total_count']

  # Identify active lifetimes
  df['active_lifetime'] = np.where((df['fraud_intensity'] > I_threshold) & (df['fraud_frequency'] > F_threshold), 1, 0)

  transaction_features = df[['transaction_features']]
  account_features = df[['account_features']]
  customer_features = df[['customer_features']]