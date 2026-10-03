from typing import NamedTuple

import numpy as np
import pandas as pd

TIME_SLOT = "30min"


class ActiveLifetime(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp


def slot_index(timestamps, time_slot=TIME_SLOT):
    if len(timestamps) == 0:
        return pd.DatetimeIndex([])
    slots = timestamps.dt.floor(time_slot)
    return pd.date_range(slots.min(), slots.max(), freq=time_slot)


def merge_active_slots(active_slots, time_slot=TIME_SLOT):
    """Merges consecutive active time slots into active lifetimes AL_1, AL_2, ..."""
    step = pd.Timedelta(time_slot)
    lifetimes = []
    for slot in sorted(active_slots):
        if lifetimes and lifetimes[-1].end == slot:
            lifetimes[-1] = ActiveLifetime(lifetimes[-1].start, slot + step)
        else:
            lifetimes.append(ActiveLifetime(slot, slot + step))
    return lifetimes


def assign_lifetimes(timestamps, lifetimes):
    """Returns the index n of the active lifetime AL_n each transaction belongs to, or -1."""
    if not lifetimes:
        return np.full(len(timestamps), -1)
    t = timestamps.to_numpy(dtype="datetime64[ns]")
    starts = np.array([lifetime.start for lifetime in lifetimes], dtype="datetime64[ns]")
    ends = np.array([lifetime.end for lifetime in lifetimes], dtype="datetime64[ns]")
    candidate = np.searchsorted(starts, t, side="right") - 1
    inside = (candidate >= 0) & (t < ends[candidate.clip(0)])
    return np.where(inside, candidate, -1)


class ActiveLifetimeIdentifier:
    """AL = {TS_j | I_fraud(TS_j) / mean(I_fraud) > I_threshold and F_fraud(TS_j) / mean(F_fraud) > F_threshold}.

    I_fraud (fraud amount / total amount) and F_fraud (fraud count / total count) are proportions
    in [0, 1]. The thresholds apply to each indicator divided by its mean over all time slots of
    the fitted period, where a slot without transactions counts as zero.
    """

    def __init__(self, intensity_threshold=1.2, frequency_threshold=4.0, time_slot=TIME_SLOT):
        self.intensity_threshold = intensity_threshold
        self.frequency_threshold = frequency_threshold
        self.time_slot = time_slot

    def fit(self, timestamps, amounts, labels):
        if len(timestamps) == 0 or timestamps.isna().any():
            raise ValueError("active lifetime identification requires valid, nonempty timestamps")
        if not np.isfinite(amounts).all() or (np.asarray(amounts) < 0).any():
            raise ValueError("transaction amounts must be finite and nonnegative")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("fraud labels must be binary")
        frame = pd.DataFrame({
            "slot": timestamps.dt.floor(self.time_slot).to_numpy(),
            "amount": np.asarray(amounts, dtype=float),
            "fraud": np.asarray(labels, dtype=int),
        })
        frame["fraud_amount"] = frame["amount"] * frame["fraud"]
        stats = (
            frame.groupby("slot")
            .agg(
                total_amount=("amount", "sum"),
                fraud_amount=("fraud_amount", "sum"),
                total_count=("fraud", "size"),
                fraud_count=("fraud", "sum"),
            )
            .reindex(slot_index(timestamps, self.time_slot), fill_value=0)
        )
        stats["intensity"] = (stats["fraud_amount"] / stats["total_amount"]).fillna(0.0)
        stats["frequency"] = (stats["fraud_count"] / stats["total_count"]).fillna(0.0)
        for metric in ("intensity", "frequency"):
            mean = stats[metric].mean()
            stats[f"relative_{metric}"] = stats[metric] / mean if mean > 0 else 0.0
        stats["active"] = (stats["relative_intensity"] > self.intensity_threshold) & (
            stats["relative_frequency"] > self.frequency_threshold
        )

        self.slot_stats_ = stats
        self.lifetimes_ = merge_active_slots(stats.index[stats["active"]], self.time_slot)
        return self

    @property
    def avg_intensity(self):
        """Avg(I_fraud): mean relative-to-mean fraud intensity over the active time slots."""
        values = self.slot_stats_.loc[self.slot_stats_["active"], "relative_intensity"]
        return float(values.mean()) if len(values) else 0.0

    @property
    def avg_frequency(self):
        """Avg(F_fraud): mean relative-to-mean fraud frequency over the active time slots."""
        values = self.slot_stats_.loc[self.slot_stats_["active"], "relative_frequency"]
        return float(values.mean()) if len(values) else 0.0
