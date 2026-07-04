"""Registries of imputer, scaler, and encoder factories.

Each registry maps a config string key (e.g. ``preprocessing.numeric.imputer:
"median"``) to a factory ``(options: dict) -> transformer``. The builder
resolves keys lazily, so adding a new component is one decorated function here
— no orchestration changes required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from ml_pipeline.core.registry import Registry
from ml_pipeline.preprocessing.transformers import FrequencyEncoder

logger = logging.getLogger(__name__)

#: Factory signature every registered component follows.
TransformerFactory = Callable[[dict[str, Any]], Any]

IMPUTER_REGISTRY: Registry[TransformerFactory] = Registry("imputer")
SCALER_REGISTRY: Registry[TransformerFactory] = Registry("scaler")
ENCODER_REGISTRY: Registry[TransformerFactory] = Registry("encoder")


# ---------------------------------------------------------------------------
# imputers
# ---------------------------------------------------------------------------


@IMPUTER_REGISTRY.register("mean")
def make_mean_imputer(options: dict[str, Any]) -> SimpleImputer:
    """Mean imputation for numeric columns."""
    return SimpleImputer(strategy="mean", **options)


@IMPUTER_REGISTRY.register("median")
def make_median_imputer(options: dict[str, Any]) -> SimpleImputer:
    """Median imputation for numeric columns (robust to outliers)."""
    return SimpleImputer(strategy="median", **options)


@IMPUTER_REGISTRY.register("most_frequent")
def make_most_frequent_imputer(options: dict[str, Any]) -> SimpleImputer:
    """Mode imputation; works for both numeric and categorical columns."""
    return SimpleImputer(strategy="most_frequent", **options)


@IMPUTER_REGISTRY.register("constant")
def make_constant_imputer(options: dict[str, Any]) -> SimpleImputer:
    """Constant-fill imputation; pass ``fill_value`` via options."""
    return SimpleImputer(strategy="constant", **options)


@IMPUTER_REGISTRY.register("knn")
def make_knn_imputer(options: dict[str, Any]) -> KNNImputer:
    """K-nearest-neighbours imputation for numeric columns."""
    return KNNImputer(**options)


# ---------------------------------------------------------------------------
# scalers
# ---------------------------------------------------------------------------


@SCALER_REGISTRY.register("standard")
def make_standard_scaler(options: dict[str, Any]) -> StandardScaler:
    """Zero-mean / unit-variance scaling."""
    return StandardScaler(**options)


@SCALER_REGISTRY.register("minmax")
def make_minmax_scaler(options: dict[str, Any]) -> MinMaxScaler:
    """Scale features into a fixed range (default ``[0, 1]``)."""
    return MinMaxScaler(**options)


@SCALER_REGISTRY.register("robust")
def make_robust_scaler(options: dict[str, Any]) -> RobustScaler:
    """Median/IQR scaling, robust to outliers."""
    return RobustScaler(**options)


@SCALER_REGISTRY.register("none")
def make_no_scaler(options: dict[str, Any]) -> str:
    """No scaling: returns ``"passthrough"``; the builder skips the step."""
    return "passthrough"


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------


@ENCODER_REGISTRY.register("onehot")
def make_onehot_encoder(options: dict[str, Any]) -> OneHotEncoder:
    """Dense one-hot encoding; unknown categories at inference encode to all-zeros."""
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **options)


@ENCODER_REGISTRY.register("ordinal")
def make_ordinal_encoder(options: dict[str, Any]) -> OrdinalEncoder:
    """Ordinal encoding; unknown categories at inference encode to ``-1``."""
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **options)


@ENCODER_REGISTRY.register("target")
def make_target_encoder(options: dict[str, Any]) -> TargetEncoder:
    """Cross-fitted target encoding (needs ``y`` at fit; ColumnTransformer forwards it)."""
    return TargetEncoder(**options)


@ENCODER_REGISTRY.register("frequency")
def make_frequency_encoder(options: dict[str, Any]) -> FrequencyEncoder:
    """Relative-frequency encoding; unseen/missing values map to ``0.0``."""
    if options:
        logger.debug("FrequencyEncoder takes no options; ignoring %s", sorted(options))
    return FrequencyEncoder()
