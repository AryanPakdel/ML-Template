"""Metric computation and primary-metric resolution."""

from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.core.types import TaskType
from ml_pipeline.evaluation.metrics import compute_metrics, resolve_primary_metric


def test_perfect_classification() -> None:
    """Identical y_true/y_pred scores 1.0 on accuracy and f1."""
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    metrics = compute_metrics(TaskType.CLASSIFICATION, y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_perfect_regression() -> None:
    """Identical y_true/y_pred gives rmse 0 and r2 1."""
    y = np.array([1.5, -2.0, 3.25, 0.0, 4.75])
    metrics = compute_metrics(TaskType.REGRESSION, y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_resolve_primary_metric_auto() -> None:
    """'auto' picks f1 (classification) and rmse (regression) with directions."""
    assert resolve_primary_metric(TaskType.CLASSIFICATION) == ("f1", "maximize")
    assert resolve_primary_metric(TaskType.REGRESSION) == ("rmse", "minimize")
    assert resolve_primary_metric(TaskType.REGRESSION, "mae") == ("mae", "minimize")


def test_unknown_metric_raises() -> None:
    """A metric key outside METRIC_DIRECTIONS is rejected."""
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_primary_metric(TaskType.CLASSIFICATION, "magic_score")
