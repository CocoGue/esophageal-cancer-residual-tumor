import os
import pandas as pd
from typing import Iterable

REQUIRED_COLUMNS = {
    "ID",
    "Num_Component",
    "Volume_Component",
    "FPC",
    "Histology",
    "TRG",
    "Residual_Disease",
    "StudyID",
    "Age",
    "Gender",
    "cT",
    "cN",
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file and validate its schema.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    data = pd.read_csv(csv_path)

    _validate_columns(data, REQUIRED_COLUMNS)

    return data


def _validate_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )
