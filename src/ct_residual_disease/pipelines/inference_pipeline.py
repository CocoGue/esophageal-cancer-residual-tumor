"""
Inference pipeline.

This module applies a trained model to new data loaded from a CSV file
and computes predictions and evaluation metrics.
"""

import numpy as np
import pandas as pd

from ct_residual_disease.data.io import load_dataset
from ct_residual_disease.data.preprocessing import (
    cast_variable_types,
    encode_for_logistic_regression,
)
from ct_residual_disease.pipelines.features_pipeline import (
    select_feature_columns,
)
from ct_residual_disease.evaluation.metrics import (
    compute_probability_metrics,
    compute_classification_metrics,
    apply_threshold,
)


def run_inference_pipeline(
    csv_path: str,
    model,
    threshold: float,
    feature_mode: str,
    encoding: str,
    drop_first: bool,
    scaler=None,
    target_column: str = "Residual_Disease",
):
    """
    Run inference on new data from a CSV file using a trained model.

    Parameters
    ----------
    csv_path : str
        Path to input CSV file.
    model
        Trained model implementing predict_proba().
    threshold : float
        Decision threshold.
    feature_mode : {"combined", "clinical_only", "volume_only"}
        Feature selection mode.
    encoding : {"logistic", "none"}
        Encoding strategy.
    drop_first : bool
        Whether to drop the first level in one-hot encoding.
    scaler : object or None
        Fitted scaler from training (if scaling was used).
    target_column : str, default="Residual_Disease"
        Name of the target column.

    Returns
    -------
    y_prob : np.ndarray
        Predicted probabilities.
    y_pred : np.ndarray
        Binary predictions.
    prob_metrics : dict
        Probability-based metrics (AUC, AUPRC).
    cls_metrics : dict
        Classification metrics (FNR, sensitivity, etc.).
    """
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = load_dataset(csv_path)

    # ------------------------------------------------------------------
    # Enforce variable types
    # ------------------------------------------------------------------
    typed_data = cast_variable_types(data)

    # ------------------------------------------------------------------
    # Extract target
    # ------------------------------------------------------------------
    if target_column not in typed_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data."
        )

    y_true = typed_data[target_column].astype(int)

    # ------------------------------------------------------------------
    # Select feature columns
    # ------------------------------------------------------------------
    X = select_feature_columns(
        data=typed_data,
        feature_mode=feature_mode,
    )

    # ------------------------------------------------------------------
    # Encode categorical variables (if required)
    # ------------------------------------------------------------------
    if encoding == "logistic" and feature_mode != "volume_only":
        X = encode_for_logistic_regression(
            X,
            drop_first=drop_first,
        )

    # ------------------------------------------------------------------
    # Apply scaling (if used during training)
    # ------------------------------------------------------------------
    if scaler is not None:
        columns_to_scale = scaler.feature_names_in_
        X[columns_to_scale] = X[columns_to_scale].astype(float)
        X.loc[:, columns_to_scale] = scaler.transform(
            X[columns_to_scale]
        )

    # ------------------------------------------------------------------
    # Predict probabilities
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X)

    # ------------------------------------------------------------------
    # Apply threshold
    # ------------------------------------------------------------------
    y_pred = apply_threshold(y_prob, threshold)

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    prob_metrics = compute_probability_metrics(
        y_true=y_true.values,
        y_prob=y_prob,
    )

    cls_metrics = compute_classification_metrics(
        y_true=y_true.values,
        y_pred=y_pred,
    )

    return prob_metrics, cls_metrics
