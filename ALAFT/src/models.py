import numpy as np
import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, BatchSampler


class ConstantBinaryClassifier:
    """Explicit fallback for a training period containing just one binary class."""

    def __init__(self, label):
        self.label = int(label)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        p = np.full(len(X), float(self.label))
        return np.column_stack([1 - p, p])


class NonSingletonBatchSampler(BatchSampler):
    """Keep every sample, merging a final singleton into the preceding batch."""

    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    return
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        length = super().__len__()
        return length - int(length > 1 and len(self.sampler) % self.batch_size == 1)


class LossWeightedTabNetClassifier(TabNetClassifier):
    """Carry sample weights in targets and apply them to cross-entropy, not sampling."""

    def _construct_loaders(self, X_train, y_train, eval_set):
        _, valid = super()._construct_loaders(X_train, y_train, eval_set)
        weights = self.loss_weights_ / self.loss_weights_.mean()
        target = np.column_stack([self.prepare_target(y_train), weights]).astype(np.float32)
        dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.from_numpy(target))
        sampler = NonSingletonBatchSampler(RandomSampler(dataset), self.batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.training_batch_count_ = len(loader)
        return loader, valid

    def compute_loss(self, y_pred, y_true):
        if y_true.ndim == 1:  # validation labels do not carry training weights
            return F.cross_entropy(y_pred, y_true.long())
        return (F.cross_entropy(y_pred, y_true[:, 0].long(), reduction="none") * y_true[:, 1]).mean()


def svm_classifier(seed=42):
    return SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)


def random_forest_classifier(seed=42):
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )


def xgboost_classifier(seed=42):
    return XGBClassifier(
        max_depth=7,
        n_estimators=100,
        learning_rate=0.12,
        min_child_weight=1,
        subsample=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


class TabNet:
    """Scikit-learn style wrapper around pytorch-tabnet with the paper's settings."""

    def __init__(self, seed=42, max_epochs=200, batch_size=512):
        self.model = LossWeightedTabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=seed, verbose=0)
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if len(y) == 0 or weights.shape != y.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("TabNet needs nonempty data and finite, nonnegative weights with positive sum")
        if not np.isin(y, [0, 1]).all() or not np.isfinite(X).all():
            raise ValueError("TabNet needs finite features and binary labels")
        keep = weights > 0
        X, y, weights = X[keep], y[keep], weights[keep]
        self.classes_ = np.array([0, 1])
        self.constant_ = ConstantBinaryClassifier(y[0]) if len(np.unique(y)) == 1 else None
        if self.constant_ is not None:
            return self
        self.model.loss_weights_ = weights
        self.model.fit(
            X,
            y,
            max_epochs=self.max_epochs,
            batch_size=min(self.batch_size, len(y)),
            virtual_batch_size=min(128, self.batch_size, len(y)),
            weights=0,
            drop_last=False,
        )
        return self

    def predict_proba(self, X):
        if self.constant_ is not None:
            return self.constant_.predict_proba(X)
        return self.model.predict_proba(X)


DETECTORS = {
    "svm": svm_classifier,
    "rf": random_forest_classifier,
    "xgboost": xgboost_classifier,
    "tabnet": TabNet,
}
