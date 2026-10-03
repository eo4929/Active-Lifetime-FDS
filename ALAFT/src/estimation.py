import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .active_lifetime import TIME_SLOT, assign_lifetimes, merge_active_slots, slot_index
from .models import ConstantBinaryClassifier, TabNet, random_forest_classifier, xgboost_classifier
from .schema import CATEGORIES, LEVELS, TIME_COLUMN, categorical_columns, numeric_columns


def slot_features(transactions, time_slot=TIME_SLOT):
    """Explanatory variables of each time slot: means of continuous and modes of categorical features."""
    slots = transactions[TIME_COLUMN].dt.floor(time_slot).rename("slot")
    numeric = [column for level in LEVELS for column in numeric_columns(level)]
    categorical = [column for level in LEVELS for column in categorical_columns(level)]

    means = transactions[numeric].replace([np.inf, -np.inf], np.nan).groupby(slots).mean()
    modes = pd.DataFrame({
        column: transactions[column].groupby(slots).agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "__unknown__")
        for column in categorical
    })
    codes = modes.apply(lambda mode: pd.Categorical(mode, categories=CATEGORIES[mode.name]).codes)
    features = pd.concat([means, codes], axis=1)
    features["Transaction_Count"] = slots.value_counts()

    features = features.reindex(slot_index(transactions[TIME_COLUMN], time_slot))
    features[categorical] = features[categorical].fillna(-1)
    features = features.fillna(0.0)
    features["Slot_Hour"] = features.index.hour + features.index.minute / 60
    features["Slot_Weekday"] = features.index.weekday
    return features


class ActiveLifetimeEstimator:
    """Classifies time slots of future transactions into active (1) and non-active (0) lifetimes.

    A bagging ensemble of RandomForest, XGBoost and TabNet is trained on the time slots of the
    current transactions, annotated with the identified active lifetimes.
    """

    def __init__(self, time_slot=TIME_SLOT, threshold=0.5, seed=42, tabnet_epochs=200):
        self.time_slot = time_slot
        self.threshold = threshold
        self.seed = seed
        self.tabnet_epochs = tabnet_epochs

    def fit(self, slots, is_active):
        if len(slots) == 0 or len(slots) != len(is_active) or not np.isin(is_active, [0, 1]).all():
            raise ValueError("AL estimation needs nonempty slots and aligned binary labels")
        if isinstance(is_active, pd.Series) and not is_active.index.equals(slots.index):
            raise ValueError("AL label index must match slot feature index")
        self.feature_names_ = list(slots.columns)
        self.scaler_ = StandardScaler().fit(slots)
        X, y = self.scaler_.transform(slots), np.asarray(is_active, dtype=int)
        self.models_, self.bootstrap_indices_ = {}, {}
        rng = np.random.default_rng(self.seed)
        factories = {"RandomForest": random_forest_classifier, "XGBoost": xgboost_classifier,
                     "TabNet": lambda seed: TabNet(seed, max_epochs=self.tabnet_epochs)}
        for i, (name, factory) in enumerate(factories.items()):
            # Stratified bootstrap preserves both labels even in sparse AL data.
            indices = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in np.unique(y)])
            rng.shuffle(indices)
            self.bootstrap_indices_[name] = indices
            model = ConstantBinaryClassifier(y[0]) if len(np.unique(y)) == 1 else factory(self.seed + i)
            self.models_[name] = model.fit(X[indices], y[indices])
        return self

    def predict_proba_by_model(self, slots):
        X = self.scaler_.transform(slots[self.feature_names_])
        return {name: model.predict_proba(X)[:, 1] for name, model in self.models_.items()}

    def predict_proba(self, slots):
        return np.mean(list(self.predict_proba_by_model(slots).values()), axis=0)

    def estimate_lifetimes(self, transactions):
        slots = slot_features(transactions, self.time_slot)
        is_active = self.predict_proba(slots) >= self.threshold
        return merge_active_slots(slots.index[is_active], self.time_slot)

    def holdout_evaluation(self, slots, is_active, train_ratio=0.8):
        """Macro-F1 and ROC-AUC of every member and the ensemble on the last (1 - train_ratio) of time slots."""
        cut = int(len(slots) * train_ratio)
        is_active = np.asarray(is_active, dtype=int)
        estimator = ActiveLifetimeEstimator(self.time_slot, self.threshold, self.seed, self.tabnet_epochs).fit(slots[:cut], is_active[:cut])

        probabilities = estimator.predict_proba_by_model(slots[cut:])
        probabilities["Ensemble"] = np.mean(list(probabilities.values()), axis=0)
        return pd.DataFrame({
            name: {
                "Macro-F1": f1_score(is_active[cut:], proba >= self.threshold, average="macro"),
                "ROC-AUC": roc_auc_score(is_active[cut:], proba),
            }
            for name, proba in probabilities.items()
        }).T


class FraudScoreAdjuster:
    """Adjusts fraud scores of transactions outside the estimated active lifetimes.

    F_adjusted(x_i) = F(x_i) * (TI_prev(x_i) + TI_next(x_i)) / 2,  TI(x_i) = exp(-rho * dT(x_i)),
    where dT is measured from the end of the previous and to the start of the next active lifetime.
    When only one of them exists, the adjusted score is F(x_i) times its temporal influence.
    """

    def __init__(self, rho=0.1, time_unit="1min"):
        if not np.isfinite(rho) or rho < 0 or pd.Timedelta(time_unit) <= pd.Timedelta(0):
            raise ValueError("rho must be finite and nonnegative, time_unit must be positive")
        self.rho = rho
        self.time_unit = time_unit

    def temporal_influence(self, distance):
        return np.exp(-self.rho * distance)

    def adjust(self, scores, timestamps, lifetimes):
        scores = np.asarray(scores, dtype=float)
        if not lifetimes:
            return scores

        t = timestamps.to_numpy(dtype="datetime64[ns]")
        starts = np.array([lifetime.start for lifetime in lifetimes], dtype="datetime64[ns]")
        ends = np.array([lifetime.end for lifetime in lifetimes], dtype="datetime64[ns]")
        unit = np.timedelta64(pd.Timedelta(self.time_unit))

        previous = np.searchsorted(ends, t, side="right") - 1
        upcoming = np.searchsorted(starts, t, side="right")
        has_previous, has_next = previous >= 0, upcoming < len(starts)
        ti_previous = self.temporal_influence(np.abs(t - ends[previous.clip(0)]) / unit)
        ti_next = self.temporal_influence(np.abs(starts[upcoming.clip(max=len(starts) - 1)] - t) / unit)

        factor = np.select(
            [has_previous & has_next, has_previous, has_next],
            [(ti_previous + ti_next) / 2, ti_previous, ti_next],
            default=1.0,
        )
        in_active_lifetime = assign_lifetimes(timestamps, lifetimes) >= 0
        return np.where(in_active_lifetime, scores, scores * factor)
