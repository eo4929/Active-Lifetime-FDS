import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .schema import CATEGORIES, LEVELS, MONETARY_COLUMNS, TIME_COLUMN, categorical_columns, numeric_columns, validate_transactions


class MultiLevelEncoder:
    """Encodes transaction, account and customer-level features into min-max scaled matrices."""

    def fit(self, df):
        validate_transactions(df)
        if len(df) == 0:
            raise ValueError("encoder training data cannot be empty")
        frames = {level: self._encode(df, level) for level in LEVELS}
        self.feature_names_ = {level: list(frame.columns) for level, frame in frames.items()}
        self.medians_ = {level: frame.median().fillna(0.0) for level, frame in frames.items()}
        self.scalers_ = {level: MinMaxScaler(clip=True).fit(frame.fillna(self.medians_[level]).to_numpy()) for level, frame in frames.items()}
        return self

    def transform(self, df):
        validate_transactions(df)
        if len(df) == 0:
            return {level: np.empty((0, len(self.feature_names_[level]))) for level in LEVELS}
        return {level: self.scalers_[level].transform(self._encode(df, level).fillna(self.medians_[level]).to_numpy()) for level in LEVELS}

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    @staticmethod
    def original(levels):
        return np.hstack([levels[level] for level in LEVELS])

    @staticmethod
    def _encode(df, level):
        numeric = df[numeric_columns(level)].astype(float).replace([np.inf, -np.inf], np.nan)
        monetary = [column for column in numeric.columns if column in MONETARY_COLUMNS]
        numeric[monetary] = np.sign(numeric[monetary]) * np.log1p(np.abs(numeric[monetary]))

        derived = pd.DataFrame(index=df.index)
        if level == "transaction":
            derived["Transaction_Hour"] = df[TIME_COLUMN].dt.hour + df[TIME_COLUMN].dt.minute / 60
            derived["Transaction_Weekday"] = df[TIME_COLUMN].dt.weekday
        if level == "account":
            derived["Account_Age_Days"] = (df[TIME_COLUMN] - df["Account_Creation_Date"]).dt.days

        one_hot = [
            pd.get_dummies(pd.Categorical(df[column].where(df[column].isin(CATEGORIES[column]), "__unknown__"), categories=[*CATEGORIES[column], "__unknown__"]), prefix=column, dtype=float)
            .set_index(df.index)
            for column in categorical_columns(level)
        ]
        return pd.concat([numeric, derived, *one_hot], axis=1)
