"""Headless diagnostic plots saved as PNG artifacts.

The Agg backend is forced at import time so plotting works in CI/servers with
no display. Every public function is failure-guarded: a broken plot logs a
warning and returns ``None`` — evaluation must never crash a training run.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ParamSpec

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  — backend must be set before pyplot
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from ml_pipeline.evaluation.metrics import confusion_matrix_frame  # noqa: E402

logger = logging.getLogger(__name__)

P = ParamSpec("P")

ArrayLike = np.ndarray | pd.Series | list

DPI = 120


def _plot_guard(func: Callable[P, Path | None]) -> Callable[P, Path | None]:
    """Wrap a plotting function so any failure logs a warning and yields None."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Path | None:
        try:
            return func(*args, **kwargs)
        except Exception:  # noqa: BLE001 — plots are best-effort by design
            logger.warning("Plot '%s' failed; skipping.", func.__name__, exc_info=True)
            plt.close("all")
            return None

    return wrapper


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Save ``fig`` as a PNG (dpi=120, tight bbox), close it, return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _positive_scores(y_proba: np.ndarray) -> np.ndarray | None:
    """Positive-class scores for binary problems, or ``None`` when not binary."""
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        return y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    return None


@_plot_guard
def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_labels: list[Any] | None,
    path: str | Path,
) -> Path | None:
    """Heatmap of the confusion matrix with per-cell counts.

    Args:
        y_true: ground-truth labels.
        y_pred: predicted labels aligned with ``y_true``.
        class_labels: original class labels (index = encoded class) or ``None``.
        path: output PNG path.

    Returns:
        The saved path, or ``None`` on failure.
    """
    frame = confusion_matrix_frame(y_true, y_pred, class_labels)
    matrix = frame.to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(frame.columns))
    ax.set_xticks(ticks, labels=[str(c) for c in frame.columns], rotation=45, ha="right")
    ax.set_yticks(ticks, labels=[str(i) for i in frame.index])
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:d}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return _save(fig, path)


@_plot_guard
def plot_roc_curve(y_true: ArrayLike, y_proba: np.ndarray, path: str | Path) -> Path | None:
    """ROC curve for binary classification (skipped with a warning otherwise).

    Args:
        y_true: binary ground-truth labels.
        y_proba: probability matrix ``(n, 2)`` or positive-class scores ``(n,)``.
        path: output PNG path.

    Returns:
        The saved path, or ``None`` when skipped or on failure.
    """
    scores = _positive_scores(y_proba)
    if scores is None:
        logger.warning("ROC curve is only plotted for binary tasks; skipping.")
        return None

    fpr, tpr, _ = roc_curve(np.asarray(y_true), scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return _save(fig, path)


@_plot_guard
def plot_pr_curve(y_true: ArrayLike, y_proba: np.ndarray, path: str | Path) -> Path | None:
    """Precision-recall curve for binary classification.

    Args:
        y_true: binary ground-truth labels.
        y_proba: probability matrix ``(n, 2)`` or positive-class scores ``(n,)``.
        path: output PNG path.

    Returns:
        The saved path, or ``None`` when skipped or on failure.
    """
    scores = _positive_scores(y_proba)
    if scores is None:
        logger.warning("PR curve is only plotted for binary tasks; skipping.")
        return None

    y_true = np.asarray(y_true)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")
    return _save(fig, path)


@_plot_guard
def plot_residuals(y_true: ArrayLike, y_pred: ArrayLike, path: str | Path) -> Path | None:
    """Residuals vs predicted scatter plus a residual histogram (two panels).

    Args:
        y_true: ground-truth values.
        y_pred: predicted values aligned with ``y_true``.
        path: output PNG path.

    Returns:
        The saved path, or ``None`` on failure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_scatter.scatter(y_pred, residuals, alpha=0.5, s=18, edgecolors="none")
    ax_scatter.axhline(0.0, linestyle="--", color="grey")
    ax_scatter.set_xlabel("Predicted")
    ax_scatter.set_ylabel("Residual (true - predicted)")
    ax_scatter.set_title("Residuals vs predicted")

    ax_hist.hist(residuals, bins=30)
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Residual distribution")
    fig.tight_layout()
    return _save(fig, path)


@_plot_guard
def plot_predicted_vs_actual(
    y_true: ArrayLike, y_pred: ArrayLike, path: str | Path
) -> Path | None:
    """Predicted vs actual scatter with a y=x reference line.

    Args:
        y_true: ground-truth values.
        y_pred: predicted values aligned with ``y_true``.
        path: output PNG path.

    Returns:
        The saved path, or ``None`` on failure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    ax.legend(loc="upper left")
    return _save(fig, path)


@_plot_guard
def plot_feature_importance(
    names: Sequence[str],
    values: np.ndarray,
    path: str | Path,
    top_n: int = 25,
) -> Path | None:
    """Horizontal bar chart of the ``top_n`` most important features.

    Args:
        names: feature names aligned with ``values``.
        values: importance per feature (magnitude ranks the bars).
        path: output PNG path.
        top_n: number of features to show, largest importance on top.

    Returns:
        The saved path, or ``None`` on failure.
    """
    values = np.asarray(values, dtype=float)
    if len(names) != len(values):
        raise ValueError(
            f"names ({len(names)}) and values ({len(values)}) must be the same length"
        )

    order = np.argsort(np.abs(values))[::-1][:top_n]
    # Reverse for barh so the largest bar renders at the top of the axis.
    plot_order = order[::-1]
    plot_names = [str(names[i]) for i in plot_order]
    plot_values = values[plot_order]

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(plot_order))))
    ax.barh(np.arange(len(plot_order)), plot_values)
    ax.set_yticks(np.arange(len(plot_order)), labels=plot_names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(plot_order)} feature importances")
    return _save(fig, path)
