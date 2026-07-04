"""Task-aware metric computation with a single source of truth for directions.

Every metric key emitted by :func:`compute_metrics` appears in
:data:`METRIC_DIRECTIONS`, so the tuner and leaderboard can optimize/rank any
of them without stage-specific knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)

from ml_pipeline.core.types import TaskType

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | pd.Series | list

#: Optimization direction for every metric key the evaluation stage can emit.
METRIC_DIRECTIONS: dict[str, str] = {
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1": "maximize",
    "roc_auc": "maximize",
    "pr_auc": "maximize",
    "mae": "minimize",
    "rmse": "minimize",
    "mape": "minimize",
    "r2": "maximize",
}

#: Default primary metric per task when config asks for "auto".
_AUTO_PRIMARY: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "f1",
    TaskType.REGRESSION: "rmse",
}


def resolve_primary_metric(task: TaskType, metric: str = "auto") -> tuple[str, str]:
    """Resolve the primary optimization metric and its direction for ``task``.

    Args:
        task: classification or regression.
        metric: a key from :data:`METRIC_DIRECTIONS`, or ``"auto"`` to pick the
            task default (f1 for classification, rmse for regression).

    Returns:
        ``(metric_name, direction)`` where direction is ``"maximize"`` or
        ``"minimize"``.

    Raises:
        ValueError: if ``metric`` is not ``"auto"`` and not a known metric key.
    """
    if metric == "auto":
        name = _AUTO_PRIMARY[task]
        return name, METRIC_DIRECTIONS[name]
    if metric not in METRIC_DIRECTIONS:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {sorted(METRIC_DIRECTIONS)}"
        )
    return metric, METRIC_DIRECTIONS[metric]


def compute_metrics(
    task: TaskType,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the standard metric suite for ``task``.

    Classification: accuracy, precision, recall, f1 (weighted averages,
    ``zero_division=0``); when ``y_proba`` is provided, also roc_auc and, for
    binary tasks, pr_auc. AUC metrics are skipped (with a warning) when they
    cannot be computed, e.g. on single-class folds.

    Regression: mae, rmse, r2, mape.

    Args:
        task: classification or regression.
        y_true: ground-truth targets (encoded labels for classification).
        y_pred: model predictions aligned with ``y_true``.
        y_proba: class-probability matrix of shape ``(n_samples, n_classes)``
            (classification only).

    Returns:
        Mapping of metric name to float value; every key appears in
        :data:`METRIC_DIRECTIONS`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == TaskType.REGRESSION:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        binary = y_proba.ndim == 1 or y_proba.shape[1] <= 2
        try:
            if binary:
                pos_scores = (
                    y_proba.ravel()
                    if y_proba.ndim == 1 or y_proba.shape[1] == 1
                    else y_proba[:, 1]
                )
                metrics["roc_auc"] = float(roc_auc_score(y_true, pos_scores))
                metrics["pr_auc"] = float(average_precision_score(y_true, pos_scores))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
                logger.debug("pr_auc skipped for multiclass problems.")
        except ValueError as exc:
            logger.warning("Skipping AUC metrics (e.g. single-class fold): %s", exc)

    return metrics


def compute_clustering_metrics(X: np.ndarray | pd.DataFrame, labels: ArrayLike) -> dict[str, float]:
    """Internal clustering quality metrics (provided for completeness).

    Args:
        X: feature matrix the clustering was fitted on.
        labels: cluster assignment per row.

    Returns:
        ``{"silhouette": ..., "davies_bouldin": ...}``, or ``{}`` when fewer
        than two clusters are present (the metrics are undefined there).
    """
    labels = np.asarray(labels)
    n_labels = len(np.unique(labels))
    if n_labels < 2:
        logger.warning(
            "Clustering metrics need at least 2 clusters, got %d; skipping.", n_labels
        )
        return {}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def confusion_matrix_frame(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_labels: list[Any] | None = None,
) -> pd.DataFrame:
    """Confusion matrix as a DataFrame with both axes labeled.

    Args:
        y_true: ground-truth labels (encoded or original).
        y_pred: predicted labels aligned with ``y_true``.
        class_labels: original class labels, index = encoded class. When the
            observed values are encoded indices, rows/columns are renamed to
            these labels; when ``None``, observed values label the axes.

    Returns:
        Square DataFrame with index named ``"true"`` and columns named
        ``"predicted"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    observed = np.unique(np.concatenate([y_true, y_pred])).tolist()

    if class_labels:
        if all(value in set(class_labels) for value in observed):
            matrix_labels: list[Any] = list(class_labels)
        else:  # values are encoded indices; display the original labels
            matrix_labels = list(range(len(class_labels)))
        display_labels = list(class_labels)
    else:
        matrix_labels = observed
        display_labels = observed

    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels)
    frame = pd.DataFrame(matrix, index=display_labels, columns=display_labels)
    frame.index.name = "true"
    frame.columns.name = "predicted"
    return frame
