"""PipelineBundle persistence: full-run round-trip plus model-level pickling."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.core.artifacts import PipelineBundle
from ml_pipeline.core.types import TaskType
from ml_pipeline.models import MODEL_REGISTRY


def test_bundle_roundtrip_predictions_stable(
    trained_bundle: tuple[Path, PipelineConfig],
    synthetic_classification_df: pd.DataFrame,
) -> None:
    """Loading the persisted bundle twice yields identical raw-input predictions."""
    bundle_path, cfg = trained_bundle
    raw = synthetic_classification_df.drop(columns=[cfg.data.target])

    bundle = PipelineBundle.load(bundle_path)
    preds = bundle.predict(raw)
    assert preds.shape == (len(raw),)
    assert set(np.unique(preds)) <= {0, 1}  # original class labels, not indices

    proba = bundle.predict_proba(raw)
    assert proba.shape == (len(raw), 2)

    # A second load (via the run directory this time) predicts identically.
    reloaded = PipelineBundle.load(bundle_path.parent)
    np.testing.assert_array_equal(preds, reloaded.predict(raw))


def test_bundle_metadata_sane(trained_bundle: tuple[Path, PipelineConfig]) -> None:
    """Metadata captures task, target, model, schema, labels, and metrics."""
    bundle_path, cfg = trained_bundle
    meta = PipelineBundle.load(bundle_path).metadata
    assert meta.task == TaskType.CLASSIFICATION.value
    assert meta.target == "label"
    assert meta.model_name == "logistic_regression"
    assert meta.feature_columns == ["num_a", "num_b", "cat_a"]
    assert [s["name"] for s in meta.raw_feature_schema] == meta.feature_columns
    assert meta.class_labels == [0, 1]
    assert "test" in meta.metrics and "accuracy" in meta.metrics["test"]
    assert (bundle_path.parent / "metadata.json").exists()


def test_mlp_model_joblib_roundtrip(tmp_path: Path) -> None:
    """A fitted MLP survives joblib dump/load with identical predictions."""
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(80, 4)).astype(np.float32), columns=list("abcd"))
    y = (X["a"].to_numpy() > 0).astype(np.int64)
    params = {"hidden_dims": [8], "max_epochs": 2, "batch_size": 32, "patience": 2}

    model = MODEL_REGISTRY.get("mlp")(params, TaskType.CLASSIFICATION, seed=0)
    model.fit(X, y)
    path = tmp_path / "mlp_model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    np.testing.assert_allclose(model.predict_proba(X), restored.predict_proba(X), atol=1e-6)
