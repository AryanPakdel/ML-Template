"""FastAPI app factory serving a trained :class:`PipelineBundle`.

The bundle is loaded once at app creation; the request model, value checks,
and dtype coercion are all derived from its metadata, so the same app code
serves any dataset/model the pipeline can train.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

from ml_pipeline.core.artifacts import BundleMetadata, PipelineBundle, resolve_bundle_path
from ml_pipeline.core.types import TaskType
from ml_pipeline.serving.schemas import (
    BatchPredictResponse,
    ModelInfo,
    PredictResponse,
    build_request_model,
    check_value_constraints,
)

logger = logging.getLogger(__name__)


def _to_python(value: Any) -> Any:
    """Cast a numpy scalar to its native python equivalent for JSON encoding."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_to_training_dtypes(df: pd.DataFrame, metadata: BundleMetadata) -> pd.DataFrame:
    """Re-align request dtypes with what the preprocessor saw at fit time.

    The request model intentionally uses simple JSON types (category -> str),
    but training coerced columns per the declared schema (e.g. category values
    may be ints). Numeric columns are cast back, and category values are mapped
    to their original ``allowed_values`` via string form.

    Args:
        df: raw request records, one row each.
        metadata: bundle metadata holding the raw feature ColumnSpec dumps.

    Returns:
        A copy of ``df`` with training-compatible dtypes/values.
    """
    out = df.copy()
    for spec in metadata.raw_feature_schema:
        name, dtype = str(spec["name"]), str(spec.get("dtype", "string"))
        if name not in out.columns:
            continue
        if dtype == "int":
            numeric = pd.to_numeric(out[name], errors="coerce")
            out[name] = numeric.astype("Int64" if spec.get("nullable") else "int64")
        elif dtype == "float":
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
        elif dtype == "bool":
            out[name] = out[name].astype("boolean" if spec.get("nullable") else "bool")
        elif dtype == "category" and spec.get("allowed_values"):
            lookup = {str(v): v for v in spec["allowed_values"]}
            out[name] = out[name].map(
                lambda v, lk=lookup: lk.get(str(v), v) if v is not None else v
            )
    return out


def _predict_frame(
    bundle: PipelineBundle, df: pd.DataFrame
) -> tuple[list[Any], list[dict[str, float]] | None]:
    """Run bundle inference on a coerced frame.

    Returns:
        ``(predictions, probabilities)`` — probabilities are ``None`` for
        regression tasks and for models without ``predict_proba``.
    """
    predictions = [_to_python(v) for v in bundle.predict(df)]
    probabilities: list[dict[str, float]] | None = None
    if bundle.metadata.task == TaskType.CLASSIFICATION.value:
        try:
            proba = np.asarray(bundle.predict_proba(df))
        except (NotImplementedError, AttributeError):
            logger.debug("Model %s has no predict_proba", bundle.metadata.model_name)
        else:
            labels = bundle.metadata.class_labels or list(range(proba.shape[1]))
            probabilities = [
                {str(label): float(p) for label, p in zip(labels, row, strict=True)}
                for row in proba
            ]
    return predictions, probabilities


def create_app(bundle_spec: str | Path = "latest", artifacts_dir: str = "artifacts") -> FastAPI:
    """Build a FastAPI app serving one trained bundle.

    Args:
        bundle_spec: ``"latest"``, a run directory, or a bundle.joblib path.
        artifacts_dir: artifacts root used to resolve ``"latest"``.

    Returns:
        A ready-to-serve FastAPI application with health, model-info,
        single-record, and batch prediction endpoints.
    """
    bundle_path = resolve_bundle_path(bundle_spec, artifacts_dir)
    bundle = PipelineBundle.load(bundle_path)
    metadata = bundle.metadata
    logger.info(
        "Serving bundle %s (model=%s, task=%s) from %s",
        metadata.run_id,
        metadata.model_name,
        metadata.task,
        bundle_path,
    )

    request_model = build_request_model(metadata)
    batch_request_model = create_model("BatchPredictRequest", records=(list[request_model], ...))

    app = FastAPI(
        title=f"ml-pipeline serving ({metadata.model_name})",
        version=metadata.package_version or "0.0.0",
    )

    def _run_inference(records: list[BaseModel]) -> tuple[list[Any], list[dict[str, float]] | None]:
        """Records -> frame -> constraint check (400) -> predict (500 on failure)."""
        df = pd.DataFrame([record.model_dump() for record in records])
        violations = check_value_constraints(df, metadata)
        if violations:
            raise HTTPException(status_code=400, detail=violations)
        try:
            return _predict_frame(bundle, _coerce_to_training_dtypes(df, metadata))
        except Exception as exc:
            logger.exception("Inference failed for %d record(s)", len(records))
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe including which bundle is loaded."""
        return {"status": "ok", "model": metadata.model_name, "run_id": metadata.run_id}

    @app.get("/model_info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        """Metadata about the served model: identity, schema, and metrics."""
        return ModelInfo(
            model_name=metadata.model_name,
            run_id=metadata.run_id,
            task=metadata.task,
            target=metadata.target,
            feature_columns=list(metadata.feature_columns),
            metrics=metadata.metrics,
        )

    def predict(record) -> PredictResponse:  # noqa: ANN001 — annotated dynamically below
        """Predict a single record; probabilities keyed by original class label."""
        predictions, probabilities = _run_inference([record])
        return PredictResponse(
            prediction=predictions[0],
            probabilities=probabilities[0] if probabilities is not None else None,
        )

    def predict_batch(body) -> BatchPredictResponse:  # noqa: ANN001 — annotated dynamically below
        """Predict a batch of records sent as ``{"records": [...]}``."""
        predictions, probabilities = _run_inference(list(body.records))
        return BatchPredictResponse(predictions=predictions, probabilities=probabilities)

    # The request models only exist inside this closure, so PEP 563 string
    # annotations could not be resolved by FastAPI — inject the classes directly.
    predict.__annotations__["record"] = request_model
    predict_batch.__annotations__["body"] = batch_request_model
    app.post("/predict", response_model=PredictResponse)(predict)
    app.post("/predict_batch", response_model=BatchPredictResponse)(predict_batch)

    return app
