"""
Reporting utilities for model evaluation.

This module formats evaluation metrics and saves them
to disk for further analysis or reporting.
"""

from typing import Dict

import pandas as pd


def metrics_to_dataframe(
    metrics: Dict[str, float],
    model_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Convert a metrics dictionary into a tidy DataFrame.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics.
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset (e.g. train, test, external).

    Returns
    -------
    pd.DataFrame
        Metrics formatted as a DataFrame.
    """
    records = []

    for metric_name, value in metrics.items():
        records.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "metric": metric_name,
                "value": value,
            }
        )

    return pd.DataFrame.from_records(records)


def save_metrics_csv(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save evaluation metrics to a CSV file.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing evaluation metrics.
    output_path : str
        Path where the CSV file will be saved.
    """
    metrics_df.to_csv(output_path, index=False)


def format_metrics_summary(
    metrics: Dict[str, float],
) -> str:
    """
    Format metrics as a human-readable text summary.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []

    for metric, value in metrics.items():
        lines.append(f"{metric}: {value:.3f}")

    return "\n".join(lines)
