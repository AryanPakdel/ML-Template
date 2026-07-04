"""Dynamic request/response schemas built from bundle metadata.

The request model is generated at app startup from the bundle's
``raw_feature_schema`` (the ``ColumnSpec`` dumps captured at training time), so
serving never hardcodes dataset column names. Value constraints (``ge``/``le``/
``allowed_values``) are re-checked manually here instead of importing the data
validation stage — serving stays self-contained on the persisted bundle.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, create_model

from ml_pipeline.core.artifacts import BundleMetadata

logger = logging.getLogger(__name__)

#: ColumnSpec dtype literal -> python type used in the generated request model.
_DTYPE_TO_PYTHON: dict[str, type] = {
    "int": int,
    "float": float,
    "string": str,
    "bool": bool,
    "datetime": str,
}


def _category_field_type(spec: dict[str, Any]) -> type:
    """Request-model type for a ``category`` column, derived from its values.

    Category columns keep whatever type the raw data holds (Titanic's ``Pclass``
    is ``[1, 2, 3]``), so the field type follows ``allowed_values``: all-int ->
    ``int``, numeric mix -> ``float``, anything else (or no declared values) ->
    ``str``.
    """
    allowed = spec.get("allowed_values") or []
    types = {type(v) for v in allowed if v is not None}
    if types == {int}:
        return int
    if types and types <= {int, float}:
        return float
    return str


def build_request_model(metadata: BundleMetadata) -> type[BaseModel]:
    """Create the ``PredictRequest`` pydantic model for one raw input record.

    Each entry of ``metadata.raw_feature_schema`` becomes one field: nullable
    columns are optional (``<type> | None = None``), non-nullable columns are
    required. Type errors in incoming payloads therefore surface as standard
    pydantic/FastAPI 422 responses.

    Args:
        metadata: bundle metadata holding the raw feature ColumnSpec dumps.

    Returns:
        A dynamically created pydantic model class named ``PredictRequest``.
    """
    fields: dict[str, Any] = {}
    for spec in metadata.raw_feature_schema:
        name = str(spec["name"])
        dtype = str(spec.get("dtype", "string"))
        if dtype == "category":
            python_type = _category_field_type(spec)
        else:
            python_type = _DTYPE_TO_PYTHON.get(dtype, str)
        if spec.get("nullable", False):
            fields[name] = (python_type | None, None)
        else:
            fields[name] = (python_type, ...)
    logger.debug("Built PredictRequest model with fields: %s", list(fields))
    return create_model("PredictRequest", **fields)


def check_value_constraints(df: pd.DataFrame, metadata: BundleMetadata) -> list[str]:
    """Check ``ge``/``le``/``allowed_values`` constraints from the bundle schema.

    Nulls are skipped (nullability is already enforced by the request model);
    ``allowed_values`` membership is compared on string form as well, because
    the request model coerces category columns to ``str`` while the training
    schema may declare non-string allowed values (e.g. ``[1, 2, 3]``).

    Args:
        df: one row per incoming record, columns named after raw features.
        metadata: bundle metadata holding the raw feature ColumnSpec dumps.

    Returns:
        Human-readable violation strings; an empty list means the input is ok.
    """
    violations: list[str] = []
    for spec in metadata.raw_feature_schema:
        name = str(spec["name"])
        if name not in df.columns:
            continue
        series = df[name]
        ge, le = spec.get("ge"), spec.get("le")
        allowed = spec.get("allowed_values")

        if ge is not None or le is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            if ge is not None:
                for idx, value in series[numeric.notna() & (numeric < ge)].items():
                    violations.append(
                        f"column '{name}', row {idx}: value {value!r} is below the minimum {ge}"
                    )
            if le is not None:
                for idx, value in series[numeric.notna() & (numeric > le)].items():
                    violations.append(
                        f"column '{name}', row {idx}: value {value!r} is above the maximum {le}"
                    )

        if allowed is not None:
            allowed_strs = {str(v) for v in allowed}
            bad = series.notna() & ~series.astype(str).isin(allowed_strs)
            for idx, value in series[bad].items():
                violations.append(
                    f"column '{name}', row {idx}: value {value!r} is not one of "
                    f"the allowed values {allowed}"
                )
    return violations


class PredictResponse(BaseModel):
    """Prediction for a single record."""

    prediction: Any
    probabilities: dict[str, float] | None = None


class BatchPredictResponse(BaseModel):
    """Predictions for a batch of records (probabilities align by position)."""

    predictions: list[Any]
    probabilities: list[dict[str, float]] | None = None


class ModelInfo(BaseModel):
    """Static facts about the served bundle."""

    model_name: str
    run_id: str
    task: str
    target: str
    feature_columns: list[str]
    metrics: dict[str, Any]
