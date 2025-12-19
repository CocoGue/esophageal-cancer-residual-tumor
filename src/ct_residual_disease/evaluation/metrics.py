"""
Metric computation utilities for binary classification.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def compute_probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics based on predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    Dict[str, float]
        Dictionary containing AUC and AUPRC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    return {
        "auc": auc(fpr, tpr),
        "auprc": auc(recall, precision),
    }


def apply_threshold(
    y_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert probabilities to binary predictions using a threshold.
    """
    return (y_prob >= threshold).astype(int)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute clinically relevant classification metrics.

    Focus is placed on false negatives rather than overall accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Binary predictions.

    Returns
    -------
    Dict[str, float]
        Dictionary containing classification metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity / Recall / True Positive Rate
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Negative Rate (critical metric)
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Specificity / True Negative Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Predictive values
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # F1-score (balance between precision and recall)
    f1 = f1_score(y_true, y_pred)

    # Matthews Correlation Coefficient (robust to imbalance)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = numerator / denominator if denominator > 0 else 0.0

    return {
        "sensitivity": sensitivity,
        "false_negative_rate": false_negative_rate,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
