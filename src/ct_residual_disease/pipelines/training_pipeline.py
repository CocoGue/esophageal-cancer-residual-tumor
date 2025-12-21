"""
Training pipeline for logistic regression.

Responsibilities:
- train / validation split
- feature scaling (fit on train, apply to validation)
- model training
- threshold selection on validation set
- metric computation
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ct_residual_disease.models.logistic import LogisticRegressionModel
from ct_residual_disease.evaluation.metrics import (
    compute_probability_metrics,
    compute_classification_metrics,
    apply_threshold,
)
from ct_residual_disease.evaluation.thresholding import (
    find_best_f1_threshold,
)


def train_logistic_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.3,
    random_state: int = 0,
    stratify: bool = True,
    scale_numerical: bool = True,
    feature_mode: str = "combined",
    add_intercept: bool = True,
    add_elasticnet_regularization: bool = True,
    reglog_regularization: float = 1.0,
) -> Dict[str, object]:
    """
    Train a logistic regression model with a validation split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (already encoded).
    y : pd.Series
        Target vector.
    test_size : float, default=0.3
        Proportion of validation set.
    random_state : int, default=0
        Random seed.
    stratify : bool, default=True
        Whether to stratify split by target.
    scale_numerical : bool, default=True
        Whether to scale numerical variables.
    feature_mode : {"combined", "clinical_only", "volume_only"}
        Feature configuration.
    add_intercept : bool, default=True
        Whether to add an intercept to the logistic model.
    add_elasticnet_regularization : bool, default=True
        Whether to apply elastic-net regularization.
    reglog_regularization : float, default=1.0
        Elastic-net mixing parameter.

    Returns
    -------
    Dict[str, object]
        Dictionary containing model, scaler, threshold, and metrics.
    """
    # ------------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------------
    stratify_y = y if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    # ------------------------------------------------------------------
    # Scaling (fit on train only)
    # ------------------------------------------------------------------
    scaler = None

    if scale_numerical:
        columns_to_scale = _get_columns_to_scale(
            X_train,
            feature_mode=feature_mode,
        )

        if columns_to_scale:
            scaler = StandardScaler()
            X_train[columns_to_scale] = X_train[columns_to_scale].astype(float)
            X_val[columns_to_scale] = X_val[columns_to_scale].astype(float)

            X_train.loc[:, columns_to_scale] = scaler.fit_transform(
                X_train[columns_to_scale]
            )
            X_val.loc[:, columns_to_scale] = scaler.transform(
                X_val[columns_to_scale]
            )

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    model = LogisticRegressionModel(
        add_intercept=add_intercept,
        regularization=(
            reglog_regularization
            if add_elasticnet_regularization
            else None
        ),
    )
    
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Validation predictions
    # ------------------------------------------------------------------
    y_val_prob = model.predict_proba(X_val)

    # ------------------------------------------------------------------
    # Threshold selection (validation only)
    # ------------------------------------------------------------------
    best_threshold = find_best_f1_threshold(
        y_true=y_val.values,
        y_prob=y_val_prob,
    )

    y_val_pred = apply_threshold(y_val_prob, best_threshold)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    prob_metrics = compute_probability_metrics(
        y_true=y_val.values,
        y_prob=y_val_prob,
    )

    cls_metrics = compute_classification_metrics(
        y_true=y_val.values,
        y_pred=y_val_pred,
    )

    # ------------------------------------------------------------------
    # Return artifacts
    # ------------------------------------------------------------------
    return {
        "model": model,
        "scaler": scaler,
        "threshold": best_threshold,
        "validation_probability_metrics": prob_metrics,
        "validation_classification_metrics": cls_metrics,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def _get_columns_to_scale(
    X: pd.DataFrame,
    *,
    feature_mode: str,
) -> list[str]:
    """
    Select numerical columns to scale based on feature mode.
    """
    columns = []

    if feature_mode == "combined":
        for col in ["Age", "Volume_Component"]:
            if col in X.columns:
                columns.append(col)

    elif feature_mode == "clinical_only":
        if "Age" in X.columns:
            columns.append("Age")

    elif feature_mode == "volume_only":
        if "Volume_Component" in X.columns:
            columns.append("Volume_Component")

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return columns
