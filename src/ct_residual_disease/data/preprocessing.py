"""
Data preprocessing utilities.

This module contains functions to:
- enforce correct variable types
- encode categorical variables for logistic regression
"""

from typing import List

import numpy as np
import pandas as pd


def cast_variable_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw variables to appropriate data types.

    This function enforces:
    - categorical encoding for clinical variables
    - ordinal ordering where clinically meaningful

    Parameters
    ----------
    data : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected variable types.
    """
    df = data.copy()

    # Histology: nominal categorical variable
    df["Histology"] = pd.Categorical(df["Histology"])

    # TRG: ordinal categorical variable
    trg_order = [1, 2, 3, 4]
    df["TRG"] = pd.Categorical(
        df["TRG"],
        categories=trg_order,
        ordered=True,
    )

    # Residual disease: binary categorical variable
    df["Residual_Disease"] = pd.Categorical(df["Residual_Disease"])

    # StudyID: keep as object/string
    df["StudyID"] = df["StudyID"].astype("object")

    # Gender: nominal categorical variable
    df["Gender"] = pd.Categorical(df["Gender"])

    # Clinical T stage (ordinal)
    # No T1, as they are out of scope
    #ct_order = ["T1","T2", "T3", "T4"]
    ct_order = ["T2", "T3", "T4"]
    df["cT"] = pd.Categorical(
        df["cT"],
        categories=ct_order,
        ordered=True,
    )

    # Clinical N stage (ordinal)
    cn_order = ["N0", "N1", "N2", "N3"]
    df["cN"] = pd.Categorical(
        df["cN"],
        categories=cn_order,
        ordered=True,
    )

    return df

def encode_for_logistic_regression(
    X: pd.DataFrame,
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Encode categorical variables for logistic regression.

    - Converts binary categorical variables to numerical form
    - Applies one-hot encoding to ordinal clinical stages

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    drop_first : bool, default=False
        Whether to drop the first category in one-hot encoding.

    Returns
    -------
    pd.DataFrame
        Encoded feature matrix.
    """
    df = X.copy()

    # Binary encoding
    df["Histology"] = np.where(
        df["Histology"] == "Adenocarcinoma",
        0,
        1,
    )

    df["Gender"] = np.where(
        df["Gender"] == "Female",
        0,
        1,
    )

    # One-hot encoding for clinical staging
    df = pd.get_dummies(
        df,
        columns=["cN"],
        drop_first=drop_first,
    )

    # Do not drop T2 to allow unseen categories in external datasets
    df = pd.get_dummies(
        df,
        columns=["cT"],
        drop_first=False,
    )

    # ------------------------------------------------------------------
    # Cast boolean dummies to int
    # ------------------------------------------------------------------
    bool_columns = df.select_dtypes(include="bool").columns
    if len(bool_columns) > 0:
        df[bool_columns] = df[bool_columns].astype(int)

    return df