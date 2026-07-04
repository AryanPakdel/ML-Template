"""Feature engineering: pruning, the all-disabled path, and PCA dimensions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_pipeline.config.schema import FeatureConfig
from ml_pipeline.core.types import TaskType
from ml_pipeline.features.builder import build_feature_pipeline
from ml_pipeline.features.transformers import CorrelationPruner


def _numeric_frame(n_rows: int = 100, n_cols: int = 6, seed: int = 3) -> pd.DataFrame:
    """Random all-numeric frame mimicking post-preprocessing output."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_rows, n_cols)),
        columns=[f"f{i}" for i in range(n_cols)],
    )


def test_correlation_pruner_drops_exact_duplicate() -> None:
    """An exact duplicate column is pruned; the original survives."""
    X = _numeric_frame(n_cols=3)
    X["f0_copy"] = X["f0"]  # |corr| == 1.0 with f0
    pruner = CorrelationPruner(threshold=0.95)
    out = pruner.fit_transform(X)
    assert pruner.dropped_columns_ == ["f0_copy"]
    assert list(out.columns) == ["f0", "f1", "f2"]
    assert list(pruner.get_feature_names_out()) == ["f0", "f1", "f2"]


def test_all_disabled_returns_none() -> None:
    """With every stage disabled (the default), no pipeline is built."""
    assert build_feature_pipeline(FeatureConfig(), TaskType.CLASSIFICATION, seed=0) is None


def test_pca_yields_requested_dims() -> None:
    """An integer n_components produces exactly that many pca_* columns."""
    cfg = FeatureConfig.model_validate({"pca": {"enabled": True, "n_components": 3}})
    pipeline = build_feature_pipeline(cfg, TaskType.REGRESSION, seed=0)
    assert pipeline is not None
    X = _numeric_frame(n_cols=6)
    out = pipeline.fit_transform(X)
    assert out.shape == (len(X), 3)
    assert list(out.columns) == ["pca_0", "pca_1", "pca_2"]
