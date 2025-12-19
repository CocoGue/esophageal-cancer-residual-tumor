"""
Input/output utilities for dataset loading.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "ID",
    "Age",
    "Gender",
    "Histology",
    "cT",
    "cN",
    "Volume_Component",
    "Residual_Disease",
    "TRG",
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file and validate its schema.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    data = pd.read_csv(csv_path)

    _validate_columns(data, REQUIRED_COLUMNS)

    return data


def _validate_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate that required columns are present in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Loaded dataset.
    required_columns : Iterable[str]
        Required column names.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )
