"""
Feature pipeline.

This module orchestrates the transformation of raw input data
into model-ready features and labels.

Responsibilities:
- load raw data
- enforce variable types
- select feature subsets
- encode categorical variables (if applicable)

Scaling is intentionally NOT handled here.
"""

from typing import Tuple

import pandas as pd

from ct_residual_disease.data.io import load_dataset
from ct_residual_disease.data.preprocessing import (
    cast_variable_types,
    encode_for_logistic_regression,
)


# ---------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------

CLINICAL_COLUMNS = [
    "Age",
    "Gender",
    "Histology",
    "cT",
    "cN",
]

VOLUME_COLUMNS = [
    "Volume_Component",
]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_features_and_labels(
    csv_path: str,
    feature_mode: str = "combined",
    encoding: str = "logistic",
    drop_first: bool = True,
    target_column: str = "Residual_Disease",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target vector y from raw data.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    feature_mode : str, default="combined"
        Feature subset to use:
        - "combined"
        - "clinical_only"
        - "volume_only"
    encoding : str, default="logistic"
        Encoding strategy:
        - "logistic": encode categorical variables
        - "none": no encoding
    drop_first : bool, default=True
        Whether to drop first category in one-hot encoding.
    target_column : str, default="Residual_Disease"
        Target variable name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    # ------------------------------------------------------------------
    # Load & type raw data
    # ------------------------------------------------------------------
    raw_data = load_dataset(csv_path)
    typed_data = cast_variable_types(raw_data)

    # ------------------------------------------------------------------
    # Extract target
    # ------------------------------------------------------------------
    if target_column not in typed_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    y = typed_data[target_column].astype(int)

    # ------------------------------------------------------------------
    # Select feature columns
    # ------------------------------------------------------------------
    X = select_feature_columns(
        data=typed_data,
        feature_mode=feature_mode,
    )

    # ------------------------------------------------------------------
    # Encode categorical variables (only if relevant)
    # ------------------------------------------------------------------
    if encoding == "logistic" and feature_mode != "volume_only":
        X = encode_for_logistic_regression(
            X,
            drop_first=drop_first,
        )
    elif encoding == "none":
        pass
    elif encoding == "logistic" and feature_mode == "volume_only":
        # No categorical variables to encode
        pass
    else:
        raise ValueError(f"Unknown encoding strategy: {encoding}")

    return X, y


def select_feature_columns(
    data: pd.DataFrame,
    feature_mode: str,
) -> pd.DataFrame:
    """
    Select feature columns according to the chosen feature mode.
    """
    if feature_mode == "combined":
        columns = CLINICAL_COLUMNS + VOLUME_COLUMNS

    elif feature_mode == "clinical_only":
        columns = CLINICAL_COLUMNS

    elif feature_mode == "volume_only":
        columns = VOLUME_COLUMNS

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required feature columns: {sorted(missing)}"
        )

    return data[columns].copy()
