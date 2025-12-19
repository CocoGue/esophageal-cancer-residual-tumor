"""
Threshold selection utilities.

We selected the cut-off value that yielded the highest F1-score
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Select the probability threshold that maximizes the F1-score.

    In case of ties, the threshold associated with the highest recall
    (i.e. lowest false negative rate) is selected.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Selected probability threshold.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob,
    )

    # Compute F1-scores safely
    f1_scores = np.zeros_like(precision)
    valid = (precision + recall) > 0
    f1_scores[valid] = (
        2.0 * precision[valid] * recall[valid]
        / (precision[valid] + recall[valid])
    )

    max_f1 = np.max(f1_scores)
    best_indices = np.where(f1_scores == max_f1)[0]

    # Tie-breaker: prefer higher recall (lower FNR)
    best_idx = best_indices[np.argmax(recall[best_indices])]

    # thresholds array is shorter by one element
    if best_idx < len(thresholds):
        return thresholds[best_idx]

    # Edge case: last precision/recall point
    return 0.5
