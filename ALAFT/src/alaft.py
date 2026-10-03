from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .active_lifetime import ActiveLifetimeIdentifier, assign_lifetimes
from .confidence import ConfidenceBasedNormalSampler
from .estimation import ActiveLifetimeEstimator, FraudScoreAdjuster, slot_features
from .feature_enrichment import ActiveLifetimeFeatureEnricher
from .fraud_possibility import FraudPossibilityCalculator
from .preprocessing import MultiLevelEncoder
from .models import ConstantBinaryClassifier
from .schema import AMOUNT_COLUMN, LABEL_COLUMN, TIME_COLUMN, validate_transactions
from .signed_graph import K_RATIOS, MINKOWSKI_P


@dataclass(frozen=True)
class ALAFTConfig:
    time_slot: str = "30min"
    intensity_threshold: float = 1.2
    frequency_threshold: float = 4.0
    alpha_t: float = 0.3
    gnn_hidden_dim: int = 32
    gnn_epochs: int = 50
    gnn_lr: float = 0.01
    n_folds: int = 5
    k_ratios: tuple = K_RATIOS
    topology_sample_size: int = 500
    graph_sample_ratio: float = 0.2
    graph_max_nodes: int = None
    times: int = 2
    theta: int = 15
    rho: float = 0.1
    seed: int = 42
    confidence_normalization: str = "paper"
    extraction_dim: int = 32
    decay_time_unit: str = "1min"
    query_batch_size: int = 1024
    estimator_epochs: int = 200
    device: str = "cpu"
    k_overrides: dict = None
    neighbor_backend: str = "approximate"
    approximation_eps: float = 0.1
    minkowski_p: float = MINKOWSKI_P

    def __post_init__(self):
        if not 0 <= self.alpha_t <= 1 or self.n_folds < 2 or self.gnn_epochs < 1 or self.estimator_epochs < 1:
            raise ValueError("invalid alpha, folds, or training epochs")
        if self.minkowski_p < 1:
            raise ValueError("the Minkowski order must be >= 1")
        if not self.k_ratios or any(not 0 < r <= 1 for r in self.k_ratios):
            raise ValueError("K ratios must be in (0, 1]")
        if not 0 < self.graph_sample_ratio <= 1:
            raise ValueError("graph_sample_ratio must be in (0, 1]")
        for size in (self.topology_sample_size, self.graph_max_nodes):
            if size is not None and size < 2:
                raise ValueError("graph sample limits must be >= 2 or None")


def possibility_weighted_training_set(features, y, remaining, possibility):
    """Training set whose weighted log-loss uses the fraud possibility p as a soft label.

    L = sum_{fraud} -log q(x) + sum_{other normal} -log(1 - q(x))
        + sum_{remaining normal} [p(x) (-log q(x)) + (1 - p(x)) (-log(1 - q(x)))],
    where q(x) is the predicted fraud score. Every fraud and every other normal transaction
    (high-confidence normals and normals outside active lifetimes) has weight 1, and each remaining
    normal transaction is learned as fraud with weight p and as normal with weight 1 - p.
    """
    kept = ~remaining
    p = np.asarray(possibility)[remaining]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("remaining normal transactions need finite fraud possibilities in [0, 1]")
    X = np.vstack([features[kept], features[remaining], features[remaining]])
    target = np.concatenate([y[kept], np.ones(len(p), dtype=int), np.zeros(len(p), dtype=int)])
    weight = np.concatenate([np.ones(kept.sum()), p, 1.0 - p])
    return X, target, weight


class ALAFT:
    """Active Lifetime-Aware Approach to Fraudulent financial Transactions on top of a base detection model."""

    def __init__(self, detector, config=ALAFTConfig()):
        self.detector = detector
        self.config = config
        self.encoder = MultiLevelEncoder()
        self.identifier = ActiveLifetimeIdentifier(config.intensity_threshold, config.frequency_threshold, config.time_slot)
        self.enricher = ActiveLifetimeFeatureEnricher(
            alpha_t=config.alpha_t,
            hidden_dim=config.gnn_hidden_dim,
            epochs=config.gnn_epochs,
            lr=config.gnn_lr,
            n_folds=config.n_folds,
            k_ratios=config.k_ratios,
            topology_sample_size=config.topology_sample_size,
            graph_sample_ratio=config.graph_sample_ratio,
            graph_max_nodes=config.graph_max_nodes,
            seed=config.seed,
            extraction_dim=config.extraction_dim,
            query_batch_size=config.query_batch_size,
            device=config.device,
            k_overrides=config.k_overrides,
            neighbor_backend=config.neighbor_backend,
            approximation_eps=config.approximation_eps,
            minkowski_p=config.minkowski_p,
        )
        self.sampler = ConfidenceBasedNormalSampler(config.times, config.confidence_normalization)
        self.possibility = FraudPossibilityCalculator(config.theta, config.query_batch_size)
        self.estimator = ActiveLifetimeEstimator(config.time_slot, seed=config.seed, tabnet_epochs=config.estimator_epochs)
        self.adjuster = FraudScoreAdjuster(config.rho, config.decay_time_unit)

    def fit(self, transactions):
        validate_transactions(transactions, require_labels=True)
        timestamps, y = transactions[TIME_COLUMN], transactions[LABEL_COLUMN].to_numpy(dtype=int)
        self.identifier.fit(timestamps, transactions[AMOUNT_COLUMN], y)
        lifetime_ids = assign_lifetimes(timestamps, self.identifier.lifetimes_)

        levels = self.encoder.fit_transform(transactions)

        def fold_context(rows):
            subset = transactions.iloc[rows]
            identifier = ActiveLifetimeIdentifier(self.config.intensity_threshold, self.config.frequency_threshold,
                                                 self.config.time_slot)
            identifier.fit(subset[TIME_COLUMN], subset[AMOUNT_COLUMN], subset[LABEL_COLUMN])
            return (assign_lifetimes(subset[TIME_COLUMN], identifier.lifetimes_) >= 0,
                    identifier.avg_intensity, identifier.avg_frequency)

        enriched = self.enricher.fit_transform(
            levels, y, lifetime_ids >= 0, self.identifier.avg_intensity, self.identifier.avg_frequency,
            fold_context=fold_context,
        )

        high_confidence = self.sampler.sample(enriched, y, lifetime_ids)
        remaining = (lifetime_ids >= 0) & (y == 0) & ~high_confidence
        possibility = self.possibility.fraud_possibility(enriched, y, timestamps, lifetime_ids, remaining)

        X, target, weight = possibility_weighted_training_set(enriched.final, y, remaining, possibility)
        positive_weight = weight > 0
        self.fitted_detector_ = (ConstantBinaryClassifier(target[positive_weight][0])
                                 if len(np.unique(target[positive_weight])) == 1 else self.detector)
        self.fitted_detector_.fit(X[positive_weight], target[positive_weight], sample_weight=weight[positive_weight])
        self.possibility_scores_ = possibility
        self.high_confidence_mask_ = high_confidence

        self.slots_ = slot_features(transactions, self.config.time_slot)
        self.estimator.fit(self.slots_, self.identifier.slot_stats_["active"])

        self.summary_ = {
            "active lifetimes": len(self.identifier.lifetimes_),
            "active time slots": int(self.identifier.slot_stats_["active"].sum()),
            "transactions in active lifetimes (SHAP sample)": self.enricher.shap_sample_size_,
            "high-confidence normal transactions": int(high_confidence.sum()),
            "remaining normal transactions": int(remaining.sum()),
            "folds actually trained": len(self.enricher.oof_splits_),
            "extracted features per level": self.config.extraction_dim,
            "mean fraud intensity / frequency ratio over training time slots": (
                float(self.identifier.slot_stats_["intensity"].mean()), float(self.identifier.slot_stats_["frequency"].mean())),
            "decay time unit": self.config.decay_time_unit,
            **{
                f"{level} graph: K ratios (+, -) / K+, K- per class": (
                    (graph.positive_ratio, graph.negative_ratio),
                    {int(c): (graph.k_pos_[c], graph.k_neg_[c]) for c in graph.classes_},
                )
                for level, graph in self.enricher.graphs_.items()
            },
        }
        return self

    def fraud_scores(self, transactions):
        validate_transactions(transactions)
        if len(transactions) == 0:
            return np.empty(0)
        enriched = self.enricher.transform(self.encoder.transform(transactions))
        return self.fitted_detector_.predict_proba(enriched.final)[:, 1]

    def adjust_fraud_scores(self, transactions, scores):
        validate_transactions(transactions)
        if len(scores) != len(transactions) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite and aligned with transactions")
        if len(transactions) == 0:
            self.estimated_lifetimes_ = []
            return np.empty(0)
        self.estimated_lifetimes_ = self.estimator.estimate_lifetimes(transactions)
        return self.adjuster.adjust(scores, transactions[TIME_COLUMN], self.estimated_lifetimes_)

    def predict_proba(self, transactions):
        scores = self.adjust_fraud_scores(transactions, self.fraud_scores(transactions))
        return np.column_stack([1 - scores, scores])

    def update(self, transactions, lookback_days=30, threshold_quantile=None):
        """Appendix L: refit on an explicitly supplied recent labeled window.

        Quantile recalibration is opt-in because the paper does not specify its
        threshold estimation rule. No scheduling or unavailable future labels.
        """
        validate_transactions(transactions, require_labels=True)
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        end = transactions[TIME_COLUMN].max()
        recent = transactions[transactions[TIME_COLUMN] > end - pd.Timedelta(days=lookback_days)].copy()
        config = self.config
        if threshold_quantile is not None:
            if not 0 < threshold_quantile < 1:
                raise ValueError("threshold_quantile must be in (0, 1)")
            probe = ActiveLifetimeIdentifier(time_slot=config.time_slot)
            probe.fit(recent[TIME_COLUMN], recent[AMOUNT_COLUMN], recent[LABEL_COLUMN])
            config = replace(config,
                             intensity_threshold=float(probe.slot_stats_.relative_intensity.quantile(threshold_quantile)),
                             frequency_threshold=float(probe.slot_stats_.relative_frequency.quantile(threshold_quantile)))
        self.__init__(self.detector, config)
        self.fit(recent)
        self.update_window_ = (recent[TIME_COLUMN].min(), recent[TIME_COLUMN].max())
        return self
