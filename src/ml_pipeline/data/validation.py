"""Config-driven schema validation: ``ColumnSpec`` list -> pandera schema -> checks.

The declared column schema in the data config is the single source of truth;
this module compiles it into a pandera ``DataFrameSchema`` and turns lazy
validation failures into one readable error instead of a wall of tracebacks.
"""

from __future__ import annotations

import logging

import pandas as pd

try:  # pandera >= 0.24 namespaces the pandas API
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - older pandera
    import pandera as pa

from ml_pipeline.config.schema import ColumnSpec, DataConfig

logger = logging.getLogger(__name__)

_MAX_EXAMPLE_VALUES = 3

# ColumnSpec.dtype literal -> pandera/pandas dtype string. Nullable ints are
# special-cased to pandas' nullable Int64 (plain int64 cannot hold NaN).
_DTYPE_MAP: dict[str, str] = {
    "int": "int64",
    "float": "float64",
    "category": "category",
    "string": "str",
    "bool": "bool",
    "datetime": "datetime64[ns]",
}


class DataValidationError(ValueError):
    """Raised when the raw dataset violates the declared column schema."""


def _pandera_dtype(spec: ColumnSpec) -> str:
    """Map a ColumnSpec dtype (plus nullability) to a pandera dtype string."""
    if spec.dtype == "int" and spec.nullable:
        return "Int64"  # pandas nullable integer; int64 cannot represent NaN
    return _DTYPE_MAP[spec.dtype]


def _build_checks(spec: ColumnSpec) -> list[pa.Check]:
    """Compile ge/le bounds and allowed_values into pandera checks."""
    checks: list[pa.Check] = []
    if spec.ge is not None:
        checks.append(pa.Check.ge(spec.ge))
    if spec.le is not None:
        checks.append(pa.Check.le(spec.le))
    if spec.allowed_values is not None:
        checks.append(pa.Check.isin(spec.allowed_values))
    return checks


def build_pandera_schema(data_cfg: DataConfig) -> pa.DataFrameSchema:
    """Build a pandera schema from the declared column specs.

    ``coerce=True`` everywhere so benign dtype drift (e.g. ints read as floats)
    is normalized rather than rejected; ``strict=False`` so undeclared columns
    pass through untouched (column roles decide their fate downstream).

    Args:
        data_cfg: validated data config with the full column schema.

    Returns:
        A ``DataFrameSchema`` ready for lazy validation.
    """
    columns: dict[str, pa.Column] = {}
    for spec in data_cfg.columns:
        columns[spec.name] = pa.Column(
            dtype=_pandera_dtype(spec),
            checks=_build_checks(spec),
            nullable=spec.nullable,
            coerce=True,
            required=True,
        )
    return pa.DataFrameSchema(columns=columns, strict=False, coerce=True)


def _summarize_failures(failure_cases: pd.DataFrame) -> str:
    """Group pandera failure cases by column & check into a readable summary."""
    lines: list[str] = []
    grouped = failure_cases.groupby(
        [failure_cases["column"].fillna("<dataframe>"), failure_cases["check"].astype(str)],
        sort=True,
    )
    for (column, check), group in grouped:
        examples = group["failure_case"].dropna().unique()[:_MAX_EXAMPLE_VALUES]
        example_str = ", ".join(repr(v) for v in examples) or "n/a"
        lines.append(
            f"  - column '{column}', check '{check}': "
            f"{len(group)} failing value(s), e.g. {example_str}"
        )
    return "\n".join(lines)


def validate_dataframe(df: pd.DataFrame, data_cfg: DataConfig) -> pd.DataFrame:
    """Validate (and coerce) the raw DataFrame against the declared schema.

    Args:
        df: raw loaded dataset.
        data_cfg: data config whose column specs define the expected schema.

    Returns:
        The validated DataFrame with coerced dtypes.

    Raises:
        DataValidationError: aggregating every schema violation in one message.
    """
    schema = build_pandera_schema(data_cfg)
    try:
        validated = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        summary = _summarize_failures(err.failure_cases)
        raise DataValidationError(
            f"Dataset failed schema validation with "
            f"{len(err.failure_cases)} failure case(s):\n{summary}"
        ) from err
    logger.info("Schema validation passed: %d rows, %d columns", *validated.shape)
    return validated
