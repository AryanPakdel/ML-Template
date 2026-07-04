"""Class-imbalance handling: sample weighting and train-fold-only resampling.

Samplers operate on the *transformed training fold only* and are never part of
the persisted inference pipeline, so there is no resampling leakage and the
serving bundle stays inference-pure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from ml_pipeline.core.registry import Registry

logger = logging.getLogger(__name__)

# Factories: (options, seed) -> imblearn sampler with fit_resample().
SAMPLER_REGISTRY: Registry[Callable[[dict[str, Any], int], Any]] = Registry("sampler")


@SAMPLER_REGISTRY.register("smote")
def _smote(options: dict[str, Any], seed: int) -> Any:
    from imblearn.over_sampling import SMOTE

    return SMOTE(random_state=seed, **options)


@SAMPLER_REGISTRY.register("random_over")
def _random_over(options: dict[str, Any], seed: int) -> Any:
    from imblearn.over_sampling import RandomOverSampler

    return RandomOverSampler(random_state=seed, **options)


@SAMPLER_REGISTRY.register("random_under")
def _random_under(options: dict[str, Any], seed: int) -> Any:
    from imblearn.under_sampling import RandomUnderSampler

    return RandomUnderSampler(random_state=seed, **options)


def apply_sampler(
    strategy: str,
    options: dict[str, Any],
    seed: int,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
    """Resample ``(X, y)`` per the configured strategy; ``none``/``class_weight`` pass through.

    Falls back to the original data (with a warning) when the sampler cannot run,
    e.g. SMOTE on a fold whose minority class is smaller than ``k_neighbors``.
    """
    if strategy in ("none", "class_weight"):
        return X, y
    sampler = SAMPLER_REGISTRY.get(strategy)(options, seed)
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info("Sampler '%s': %d -> %d training rows", strategy, len(y), len(y_res))
        return X_res, np.asarray(y_res)
    except ValueError as err:
        logger.warning("Sampler '%s' failed (%s); training on original data", strategy, err)
        return X, y


def balanced_sample_weight(strategy: str, y: np.ndarray) -> np.ndarray | None:
    """Balanced per-sample weights for strategy ``class_weight``; else ``None``."""
    if strategy != "class_weight":
        return None
    return compute_sample_weight("balanced", y)
