import numpy as np
import pandas as pd
from scipy.special import expit


def temporal_weight(minutes_apart):
    """delta_ij = 1 / (1 + e^{|t(x_i) - t(f_j)|})"""
    return expit(-np.abs(minutes_apart))


class FraudPossibilityCalculator:
    """Idea 3: fraud possibility of the remaining normal transactions in active lifetime.

    fps(x_i) = sum_{j in F} M(x_i, f_j),  M = M_o + M_t + M_a + M_c,
    M_v(x_i, f_j) = delta_ij * sum_g x_i . f^avg_{j+g} / sum_k delta_kj * sum_g x_k . f^avg_{j+g},
    where g slides from -theta to theta minutes around f_j.
    """

    def __init__(self, theta=15, batch_size=1024):
        if not isinstance(theta, int) or theta < 0 or batch_size < 1:
            raise ValueError("theta must be a nonnegative integer; batch_size must be positive")
        self.theta = theta
        self.batch_size = batch_size

    def windowed_fraud_means(self, fraud_vectors, fraud_minutes):
        """sum_{g=-theta}^{theta} f^avg_{j+g}: mean fraud vector between f_j and f_j + g minutes."""
        total = np.zeros_like(fraud_vectors, dtype=float)
        order = np.argsort(fraud_minutes, kind="stable")
        times = fraud_minutes[order]
        prefix = np.vstack([np.zeros((1, fraud_vectors.shape[1])), np.cumsum(fraud_vectors[order], axis=0, dtype=float)])
        for g in range(-self.theta, self.theta + 1):
            lower = fraud_minutes + min(g, 0)
            upper = fraud_minutes + max(g, 0)
            left = np.searchsorted(times, lower, side="left")
            right = np.searchsorted(times, upper, side="right")
            total += (prefix[right] - prefix[left]) / (right - left)[:, None]
        return total

    def correlation_scores(self, view, normal_minutes, fraud_minutes, normal_rows, fraud_rows):
        """Same column-normalized correlation sum, with bounded pair matrices."""
        windows = self.windowed_fraud_means(view[fraud_rows], fraud_minutes)
        scores = np.zeros(len(normal_rows))
        for fstart in range(0, len(fraud_rows), self.batch_size):
            ftime = fraud_minutes[fstart:fstart + self.batch_size]
            vectors = windows[fstart:fstart + self.batch_size]
            denominator = np.zeros(len(ftime))
            for nstart in range(0, len(normal_rows), self.batch_size):
                rows = normal_rows[nstart:nstart + self.batch_size]
                delta = temporal_weight(normal_minutes[nstart:nstart + len(rows), None] - ftime[None, :])
                denominator += (delta * (view[rows] @ vectors.T)).sum(axis=0)
            for nstart in range(0, len(normal_rows), self.batch_size):
                rows = normal_rows[nstart:nstart + self.batch_size]
                delta = temporal_weight(normal_minutes[nstart:nstart + len(rows), None] - ftime[None, :])
                weighted = delta * (view[rows] @ vectors.T)
                scores[nstart:nstart + len(rows)] += np.divide(weighted, denominator, out=np.zeros_like(weighted), where=denominator > 0).sum(axis=1)
        return scores

    def temporal_correlation(self, view, normal_minutes, fraud_minutes, normal_rows, fraud_rows):
        """M_v(x_k, f_j) for every normal x_k and fraud f_j of an active lifetime."""
        windowed = self.windowed_fraud_means(view[fraud_rows], fraud_minutes)
        delta = temporal_weight(normal_minutes[:, None] - fraud_minutes[None, :])
        weighted = delta * (view[normal_rows] @ windowed.T)
        denominator = weighted.sum(axis=0)
        return np.divide(weighted, denominator, out=np.zeros_like(weighted), where=denominator > 0)

    def fraud_possibility_scores(self, enriched, y, timestamps, lifetime_ids, remaining):
        """fps for each remaining normal transaction, NaN elsewhere."""
        y = np.asarray(y)
        views = enriched.views()
        minutes = ((timestamps - timestamps.min()) / pd.Timedelta(minutes=1)).to_numpy()
        scores = np.full(len(y), np.nan)

        for lifetime in np.unique(lifetime_ids[remaining]):
            in_lifetime = lifetime_ids == lifetime
            normal_rows = np.flatnonzero(in_lifetime & (y == 0))
            fraud_rows = np.flatnonzero(in_lifetime & (y == 1))
            if len(fraud_rows) == 0:
                continue

            fps = sum(
                self.correlation_scores(view, minutes[normal_rows], minutes[fraud_rows], normal_rows, fraud_rows)
                for view in views.values()
            )
            is_remaining = remaining[normal_rows]
            scores[normal_rows[is_remaining]] = fps[is_remaining]
        return scores

    def fraud_possibility(self, enriched, y, timestamps, lifetime_ids, remaining):
        """Fraud possibility in [0, 1]: min-max normalized fps of the remaining normal transactions."""
        scores = self.fraud_possibility_scores(enriched, y, timestamps, lifetime_ids, remaining)
        self.raw_scores_ = scores.copy()
        known = ~np.isnan(scores)
        if not known.any():
            return scores
        low, high = scores[known].min(), scores[known].max()
        scores[known] = (scores[known] - low) / (high - low) if high > low else (0.5 if high > 0 else 0.0)
        return scores
