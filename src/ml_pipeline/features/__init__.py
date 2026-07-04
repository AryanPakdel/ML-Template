"""Feature engineering/selection stage.

Runs on the post-preprocessing matrix (all numeric, pandas DataFrames) and is
fully config-gated: polynomial expansion, correlation pruning, importance-based
selection, and PCA are each optional. :func:`build_feature_pipeline` returns the
assembled sklearn ``Pipeline`` — or ``None`` when every step is disabled.
"""

from __future__ import annotations

from ml_pipeline.features.builder import build_feature_pipeline

__all__ = ["build_feature_pipeline"]
