import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ConfidenceBasedNormalSampler:
    """Idea 2: confidence-based normal sampling in active lifetime.

    Confidence_ALn(x_i) = 1 - log[sum_j sim_o(x_i, f_j) + LS_ij] / log[sum_m sum_j sim_o(x_m, f_j) + LS_mj],
    LS_ij = sum_j sim_t(x_i, f_j) + sim_a(x_i, f_j) + sim_c(x_i, f_j),
    and the (times x |F|) most confident normal transactions of every AL_n are sampled.
    """

    def __init__(self, times=2, normalization="paper"):
        if times < 0 or not isinstance(times, int):
            raise ValueError("times must be a nonnegative integer")
        if normalization not in {"paper", "log1p"}:
            raise ValueError("confidence normalization must be paper or log1p")
        self.times = times
        self.normalization = normalization

    @staticmethod
    def similarity_to_frauds(views, normal_rows, fraud_rows):
        """sum_j sim_o(x_i, f_j) + LS_ij for every normal transaction x_i."""
        return sum(
            cosine_similarity(view[normal_rows], view[fraud_rows]).sum(axis=1)
            for view in views.values()
        )

    def confidence(self, views, normal_rows, fraud_rows):
        similarity = np.maximum(self.similarity_to_frauds(views, normal_rows, fraud_rows), 0.0)
        total = similarity.sum()
        if total == 0:
            return np.ones_like(similarity)
        if self.normalization == "paper" and total > 1 + 1e-12:
            return 1.0 - np.log(np.maximum(similarity, np.finfo(float).tiny)) / np.log(total)
        # The literal formula is singular at total=1 and reverses the intended
        # ordering below 1. Use the documented stable extension only there.
        return 1.0 - np.log1p(similarity) / np.log1p(total)

    def sample(self, enriched, y, lifetime_ids):
        """Returns a mask of the high-confidence normal transactions within active lifetimes."""
        y = np.asarray(y)
        views = enriched.views()
        high_confidence = np.zeros(len(y), dtype=bool)

        for lifetime in np.unique(lifetime_ids[lifetime_ids >= 0]):
            in_lifetime = lifetime_ids == lifetime
            normal_rows = np.flatnonzero(in_lifetime & (y == 0))
            fraud_rows = np.flatnonzero(in_lifetime & (y == 1))
            if len(normal_rows) == 0 or len(fraud_rows) == 0:
                continue

            confidence = self.confidence(views, normal_rows, fraud_rows)
            most_confident = np.argsort(-confidence, kind="stable")[: self.times * len(fraud_rows)]
            high_confidence[normal_rows[most_confident]] = True
        return high_confidence
