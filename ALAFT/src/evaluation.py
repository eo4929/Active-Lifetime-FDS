import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from .schema import TIME_COLUMN

TOP_PERCENTS = (1, 2, 5, 10)
TEST_START = "2022-01-01"


def temporal_split(transactions, test_start=TEST_START):
    """Transactions before `test_start` for training and from `test_start` on for testing.

    With the default boundary, training covers transactions up to December 2021 and testing
    covers transactions from January 2022, as in the paper (Jan 2018-Dec 2021 / Jan-Dec 2022).
    """
    is_test = transactions[TIME_COLUMN] >= pd.Timestamp(test_start)
    if is_test.all() or not is_test.any():
        raise ValueError(f"no transactions on both sides of the split boundary {test_start}")
    return transactions[~is_test].reset_index(drop=True), transactions[is_test].reset_index(drop=True)


def top_n_predictions(scores, percent):
    predictions = np.zeros(len(scores), dtype=int)
    predictions[np.argsort(-scores, kind="stable")[: int(np.ceil(len(scores) * percent / 100))]] = 1
    return predictions


def evaluate(y, scores, percents=TOP_PERCENTS):
    y, scores = np.asarray(y), np.asarray(scores)
    result = {"PR-AUC": average_precision_score(y, scores)}
    for percent in percents:
        predictions = top_n_predictions(scores, percent)
        tn, fp, _, _ = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
        result[f"Macro-F1@{percent}%"] = f1_score(y, predictions, average="macro")
        result[f"FPR@{percent}%"] = fp / (fp + tn)
    return pd.Series(result)
