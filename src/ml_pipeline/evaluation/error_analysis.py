"""Worst-prediction extraction for manual error review.

Works on the *raw* (pre-transform) rows so a human sees the original feature
values, with ground truth, prediction, and an error score attached.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.core.types import TaskType

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | pd.Series | list

TRUE_COLUMN = "y_true"
PRED_COLUMN = "y_pred"
ERROR_COLUMN = "error"


def worst_rows(
    raw_df: pd.DataFrame,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    task: TaskType,
    y_proba: np.ndarray | None = None,
    top_n: int = 20,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return the ``top_n`` rows the model got most wrong.

    Regression: every row scored by absolute residual. Classification: only
    misclassified rows, scored by ``1 - P(predicted class)`` when ``y_proba``
    is available (encoded predictions index the probability columns), else a
    constant ``1.0``.

    Args:
        raw_df: raw feature rows, positionally aligned with the predictions.
        y_true: ground-truth targets (encoded labels for classification).
        y_pred: predictions aligned with ``y_true``.
        task: classification or regression.
        y_proba: class-probability matrix ``(n_samples, n_classes)`` or ``None``.
        top_n: number of worst rows to keep.
        id_columns: raw columns to move to the front of the result (for
            traceability, e.g. a passenger id).

    Returns:
        DataFrame with ``y_true``, ``y_pred`` and ``error`` columns attached,
        sorted by ``error`` descending, ``id_columns`` first when given.

    Raises:
        ValueError: when ``y_true``/``y_pred`` lengths do not match ``raw_df``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(raw_df) or len(y_pred) != len(raw_df):
        raise ValueError(
            f"Length mismatch: raw_df={len(raw_df)}, "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    frame = raw_df.reset_index(drop=True).copy()
    frame[TRUE_COLUMN] = y_true
    frame[PRED_COLUMN] = y_pred

    if task == TaskType.REGRESSION:
        frame[ERROR_COLUMN] = np.abs(
            y_true.astype(float) - y_pred.astype(float)
        )
    else:
        frame = frame.loc[np.asarray(y_true != y_pred)].copy()
        frame[ERROR_COLUMN] = 1.0
        if y_proba is not None and not frame.empty:
            try:
                proba = np.asarray(y_proba)
                rows = frame.index.to_numpy()
                predicted_class = y_pred[rows].astype(int)
                frame[ERROR_COLUMN] = 1.0 - proba[rows, predicted_class]
            except (IndexError, TypeError, ValueError) as exc:
                logger.warning(
                    "Could not score misclassifications from y_proba (%s); "
                    "falling back to error=1.0.",
                    exc,
                )
        logger.info("Error analysis: %d misclassified rows found.", len(frame))

    frame = frame.sort_values(ERROR_COLUMN, ascending=False).head(top_n)

    if id_columns:
        present = [c for c in id_columns if c in frame.columns]
        missing = [c for c in id_columns if c not in frame.columns]
        if missing:
            logger.warning("id_columns not found in raw data, ignoring: %s", missing)
        frame = frame[present + [c for c in frame.columns if c not in present]]

    return frame.reset_index(drop=True)


def save_error_analysis(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the error-analysis frame to CSV (parents created) and return the path.

    Args:
        df: output of :func:`worst_rows`.
        path: destination ``.csv`` path.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved error analysis (%d rows) to %s", len(df), path)
    return path
