"""Assemble the persisted preprocessing pipeline from validated config.

The builder groups feature columns by their **declared** dtype (from
``data.columns``), constructs one sub-pipeline per column group, and wires
everything into a single dense ``ColumnTransformer`` with pandas output. The
returned object is fitted by the trainer and persisted inside the
:class:`~ml_pipeline.core.artifacts.PipelineBundle`, so exactly the same
transformations run at inference time.

Row-level operations (e.g. ``outliers.method="remove"``) never belong here:
they are applied by the trainer to the training split only.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml_pipeline.config.schema import (
    ColumnOverride,
    DataConfig,
    OutlierConfig,
    PreprocessingConfig,
)
from ml_pipeline.preprocessing.components import (
    ENCODER_REGISTRY,
    IMPUTER_REGISTRY,
    SCALER_REGISTRY,
)
from ml_pipeline.preprocessing.transformers import DatetimeFeatureExtractor, OutlierClipper

logger = logging.getLogger(__name__)

_NUMERIC_DTYPES = frozenset({"int", "float"})
_CATEGORICAL_DTYPES = frozenset({"category", "string", "bool"})
_DATETIME_DTYPES = frozenset({"datetime"})

_PASSTHROUGH = "passthrough"


def _entry_name(prefix: str, index: int, column: str) -> str:
    """Unique, readable ColumnTransformer entry name (no ``__`` allowed)."""
    return f"{prefix}_{index}_{column.replace('__', '_')}"


def _effective(
    override_key: str | None,
    override_options: dict[str, Any],
    default_key: str,
    default_options: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve a (key, options) pair: an override replaces both key *and* options."""
    if override_key is not None:
        return override_key, dict(override_options)
    return default_key, dict(default_options)


def _group_signature(*parts: Any) -> str:
    """Stable hashable signature for grouping identical component choices."""
    return json.dumps(parts, sort_keys=True, default=str)


def _build_numeric_pipeline(
    imputer: str,
    imputer_options: dict[str, Any],
    scaler: str,
    scaler_options: dict[str, Any],
    outliers: OutlierConfig,
) -> Pipeline:
    """Numeric sub-pipeline: imputer -> optional IQR clipper -> optional scaler."""
    steps: list[tuple[str, Any]] = [
        ("imputer", IMPUTER_REGISTRY.get(imputer)(imputer_options))
    ]
    if outliers.method == "clip":
        steps.append(("outlier_clipper", OutlierClipper(iqr_factor=outliers.iqr_factor)))
    scaler_obj = SCALER_REGISTRY.get(scaler)(scaler_options)
    if scaler_obj != _PASSTHROUGH:
        steps.append(("scaler", scaler_obj))
    return Pipeline(steps)


def _build_categorical_pipeline(
    imputer: str,
    imputer_options: dict[str, Any],
    encoder: str,
    encoder_options: dict[str, Any],
) -> Pipeline:
    """Categorical sub-pipeline: imputer -> encoder (always dense output)."""
    return Pipeline(
        [
            ("imputer", IMPUTER_REGISTRY.get(imputer)(imputer_options)),
            ("encoder", ENCODER_REGISTRY.get(encoder)(encoder_options)),
        ]
    )


def _build_datetime_pipeline(
    parts: list[str],
    scaler: str,
    scaler_options: dict[str, Any],
) -> Pipeline:
    """Datetime sub-pipeline: part extraction -> median impute -> optional scaler."""
    steps: list[tuple[str, Any]] = [
        ("extract", DatetimeFeatureExtractor(parts=list(parts))),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    scaler_obj = SCALER_REGISTRY.get(scaler)(scaler_options)
    if scaler_obj != _PASSTHROUGH:
        steps.append(("scaler", scaler_obj))
    return Pipeline(steps)


def _resolve_categorical_choice(
    column: str,
    override: ColumnOverride | None,
    prep_cfg: PreprocessingConfig,
    train_df: pd.DataFrame | None,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Effective (imputer, options, encoder, options) for one categorical column.

    Applies the cardinality guard: when a training frame is available and the
    column slated for one-hot encoding has more unique values than
    ``categorical.max_onehot_cardinality``, the encoder degrades to
    ``"frequency"`` with a logged warning (avoids feature explosion).
    """
    cat_cfg = prep_cfg.categorical
    imputer, imputer_opts = _effective(
        override.imputer if override else None,
        override.imputer_options if override else {},
        cat_cfg.imputer,
        cat_cfg.imputer_options,
    )
    encoder, encoder_opts = _effective(
        override.encoder if override else None,
        override.encoder_options if override else {},
        cat_cfg.encoder,
        cat_cfg.encoder_options,
    )

    if encoder == "onehot" and train_df is not None and column in train_df.columns:
        cardinality = int(train_df[column].nunique())
        if cardinality > cat_cfg.max_onehot_cardinality:
            logger.warning(
                "Column '%s' has %d unique values in the training data "
                "(> max_onehot_cardinality=%d); switching encoder 'onehot' -> 'frequency'.",
                column,
                cardinality,
                cat_cfg.max_onehot_cardinality,
            )
            encoder, encoder_opts = "frequency", {}

    return imputer, imputer_opts, encoder, encoder_opts


def build_preprocessor(
    prep_cfg: PreprocessingConfig,
    data_cfg: DataConfig,
    train_df: pd.DataFrame | None = None,
) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline for the declared feature columns.

    Columns are grouped by declared dtype — ``int``/``float`` -> numeric,
    ``category``/``string``/``bool`` -> categorical, ``datetime`` -> datetime —
    and each group gets a sub-pipeline assembled from the component registries.
    Non-overridden columns sharing identical component choices share one
    ``ColumnTransformer`` entry; every column in ``prep_cfg.column_overrides``
    gets its own entry.

    Args:
        prep_cfg: preprocessing stage configuration.
        data_cfg: dataset definition supplying the column schema.
        train_df: optional raw training split, used only for the one-hot
            cardinality guard (no fitting happens here).

    Returns:
        An unfitted ``Pipeline`` wrapping a dense ``ColumnTransformer`` with
        ``remainder="drop"`` and pandas output.

    Raises:
        ValueError: when ``column_overrides`` names unknown feature columns, or
            no feature columns are declared.
        KeyError: when a config key is missing from a component registry.
    """
    features = data_cfg.feature_columns()
    feature_names = [spec.name for spec in features]

    unknown = sorted(set(prep_cfg.column_overrides) - set(feature_names))
    if unknown:
        raise ValueError(
            f"preprocessing.column_overrides references unknown feature columns: "
            f"{unknown}. Known feature columns: {feature_names}"
        )

    numeric_cols = [s.name for s in features if s.dtype in _NUMERIC_DTYPES]
    categorical_cols = [s.name for s in features if s.dtype in _CATEGORICAL_DTYPES]
    datetime_cols = [s.name for s in features if s.dtype in _DATETIME_DTYPES]

    overrides = prep_cfg.column_overrides
    entries: list[tuple[str, Any, list[str]]] = []

    # ----------------------------------------------------------- numeric
    base_numeric = [c for c in numeric_cols if c not in overrides]
    if base_numeric:
        entries.append(
            (
                "numeric",
                _build_numeric_pipeline(
                    prep_cfg.numeric.imputer,
                    prep_cfg.numeric.imputer_options,
                    prep_cfg.numeric.scaler,
                    prep_cfg.numeric.scaler_options,
                    prep_cfg.numeric.outliers,
                ),
                base_numeric,
            )
        )
    for index, column in enumerate(c for c in numeric_cols if c in overrides):
        override = overrides[column]
        if override.encoder is not None:
            logger.warning(
                "Ignoring encoder override for numeric column '%s' "
                "(encoders apply to categorical columns only).",
                column,
            )
        imputer, imputer_opts = _effective(
            override.imputer,
            override.imputer_options,
            prep_cfg.numeric.imputer,
            prep_cfg.numeric.imputer_options,
        )
        scaler, scaler_opts = _effective(
            override.scaler,
            override.scaler_options,
            prep_cfg.numeric.scaler,
            prep_cfg.numeric.scaler_options,
        )
        entries.append(
            (
                _entry_name("numeric_override", index, column),
                _build_numeric_pipeline(
                    imputer, imputer_opts, scaler, scaler_opts, prep_cfg.numeric.outliers
                ),
                [column],
            )
        )

    # ------------------------------------------------------- categorical
    grouped: dict[str, tuple[str, dict[str, Any], str, dict[str, Any], list[str]]] = {}
    override_index = 0
    for column in categorical_cols:
        override = overrides.get(column)
        imputer, imputer_opts, encoder, encoder_opts = _resolve_categorical_choice(
            column, override, prep_cfg, train_df
        )
        if override is not None:
            if override.scaler is not None:
                logger.warning(
                    "Ignoring scaler override for categorical column '%s' "
                    "(scalers apply to numeric columns only).",
                    column,
                )
            entries.append(
                (
                    _entry_name("categorical_override", override_index, column),
                    _build_categorical_pipeline(imputer, imputer_opts, encoder, encoder_opts),
                    [column],
                )
            )
            override_index += 1
            continue
        signature = _group_signature(imputer, imputer_opts, encoder, encoder_opts)
        group = grouped.setdefault(
            signature, (imputer, imputer_opts, encoder, encoder_opts, [])
        )
        group[4].append(column)

    for index, (imputer, imputer_opts, encoder, encoder_opts, columns) in enumerate(
        grouped.values()
    ):
        entries.append(
            (
                f"categorical_{index}_{encoder}",
                _build_categorical_pipeline(imputer, imputer_opts, encoder, encoder_opts),
                columns,
            )
        )

    # ---------------------------------------------------------- datetime
    base_datetime = [c for c in datetime_cols if c not in overrides]
    if base_datetime:
        entries.append(
            (
                "datetime",
                _build_datetime_pipeline(
                    list(prep_cfg.datetime.extract),
                    prep_cfg.numeric.scaler,
                    prep_cfg.numeric.scaler_options,
                ),
                base_datetime,
            )
        )
    for index, column in enumerate(c for c in datetime_cols if c in overrides):
        override = overrides[column]
        if override.imputer is not None or override.encoder is not None:
            logger.warning(
                "Datetime column '%s' only supports a scaler override; "
                "ignoring imputer/encoder overrides.",
                column,
            )
        scaler, scaler_opts = _effective(
            override.scaler,
            override.scaler_options,
            prep_cfg.numeric.scaler,
            prep_cfg.numeric.scaler_options,
        )
        entries.append(
            (
                _entry_name("datetime_override", index, column),
                _build_datetime_pipeline(list(prep_cfg.datetime.extract), scaler, scaler_opts),
                [column],
            )
        )

    if not entries:
        raise ValueError(
            "No feature columns produced preprocessing entries; check data.columns "
            "roles and dtypes."
        )

    logger.info(
        "Built preprocessor: %d numeric, %d categorical, %d datetime columns "
        "across %d ColumnTransformer entries.",
        len(numeric_cols),
        len(categorical_cols),
        len(datetime_cols),
        len(entries),
    )

    column_transformer = ColumnTransformer(
        transformers=entries,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline([("preprocess", column_transformer)])
    pipeline.set_output(transform="pandas")
    return pipeline
