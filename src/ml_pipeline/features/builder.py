"""Assemble the config-gated feature engineering/selection pipeline.

The resulting sklearn ``Pipeline`` runs on the post-preprocessing matrix (all
numeric, pandas in/out) and is persisted inside the
:class:`~ml_pipeline.core.artifacts.PipelineBundle`, so training and inference
always apply identical feature transformations.
"""

from __future__ import annotations

import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ml_pipeline.config.schema import FeatureConfig
from ml_pipeline.core.types import TaskType
from ml_pipeline.features.transformers import (
    CorrelationPruner,
    ImportanceSelector,
    LoggingPCA,
)

logger = logging.getLogger(__name__)


def build_feature_pipeline(
    feat_cfg: FeatureConfig, task: TaskType, seed: int
) -> Pipeline | None:
    """Build the feature engineering/selection pipeline from config.

    Steps are appended in a fixed order, each gated by its ``enabled`` flag:

    1. ``polynomial`` — :class:`~sklearn.preprocessing.PolynomialFeatures`.
       **Warning:** polynomial expansion explodes dimensionality combinatorially
       (hundreds of columns from dozens), which is why it is off by default;
       pair it with pruning/selection/PCA when you do enable it.
    2. ``correlation_pruning`` — :class:`CorrelationPruner` drops near-duplicate
       features.
    3. ``importance_selection`` — :class:`ImportanceSelector` keeps the features
       a quick random forest ranks highest (needs ``y``, which the sklearn
       Pipeline protocol forwards at fit time).
    4. ``pca`` — :class:`LoggingPCA` for dimensionality reduction with
       explained-variance logging.

    Args:
        feat_cfg: validated feature-stage config.
        task: classification or regression (drives the importance ranker).
        seed: random seed for the stochastic steps (forest, PCA solver).

    Returns:
        A pandas-output ``Pipeline`` of the enabled steps, or ``None`` when no
        step is enabled (the orchestrator then skips this stage entirely).
    """
    steps: list[tuple[str, object]] = []

    if feat_cfg.polynomial.enabled:
        steps.append(
            (
                "polynomial",
                PolynomialFeatures(
                    degree=feat_cfg.polynomial.degree,
                    interaction_only=feat_cfg.polynomial.interaction_only,
                    include_bias=False,
                ),
            )
        )

    if feat_cfg.correlation_pruning.enabled:
        steps.append(
            (
                "correlation_pruning",
                CorrelationPruner(threshold=feat_cfg.correlation_pruning.threshold),
            )
        )

    if feat_cfg.importance_selection.enabled:
        steps.append(
            (
                "importance_selection",
                ImportanceSelector(
                    task=task,
                    top_k=feat_cfg.importance_selection.top_k,
                    n_estimators=feat_cfg.importance_selection.n_estimators,
                    seed=seed,
                ),
            )
        )

    if feat_cfg.pca.enabled:
        steps.append(
            (
                "pca",
                LoggingPCA(n_components=feat_cfg.pca.n_components, random_state=seed),
            )
        )

    if not steps:
        logger.info("Feature engineering: no steps enabled; stage will be skipped.")
        return None

    logger.info("Feature pipeline steps: %s", [name for name, _ in steps])
    return Pipeline(steps).set_output(transform="pandas")
