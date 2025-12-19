"""
Feature pipeline.

This module orchestrates the transformation of raw input data
into model-ready features and labels.

Responsibilities:
- load raw data
- enforce variable types
- encode features
- split features and target

This pipeline contains no model logic.
"""

from typing import Tuple

import pandas as pd

from ct_residual_disease.data.io import load_dataset
from ct_residual_disease.data.preprocessing import (
    cast_variable_types,
    encode_for_logistic_regression,
)


def build_features_and_labels(
    csv_path: str,
    target_column: str = "Residual_Disease",
    encoding: str = "logistic",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y from raw data.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    target_column : str, default="Residual_Disease"
        Name of the target column.
    encoding : {"logistic", "none"}, default="logistic"
        Encoding strategy to apply:
        - "logistic": one-hot / binary encoding for linear models
        - "none": no additional encoding (e.g. for tree-based models)

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    raw_data = load_dataset(csv_path)

    # ------------------------------------------------------------------
    # Enforce variable types
    # ------------------------------------------------------------------
    typed_data = cast_variable_types(raw_data)

    # ------------------------------------------------------------------
    # Separate target and features
    # ------------------------------------------------------------------
    if target_column not in typed_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    y = typed_data[target_column].astype(int)
    X = typed_data.drop(columns=[target_column])

    # ------------------------------------------------------------------
    # Encode features (model-dependent)
    # ------------------------------------------------------------------
    if encoding == "logistic":
        X = encode_for_logistic_regression(X)
    elif encoding == "none":
        # Tree-based models can handle categorical features directly
        pass
    else:
        raise ValueError(
            f"Unknown encoding strategy: {encoding}"
        )

    return X, y
