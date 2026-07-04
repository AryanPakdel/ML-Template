"""Tiny-budget MLP smoke tests: classification and regression on CPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.core.types import TaskType
from ml_pipeline.models import MODEL_REGISTRY

#: Seconds-scale training budget for CI-friendly runs.
MLP_PARAMS: dict = {"hidden_dims": [8], "max_epochs": 2, "batch_size": 32, "patience": 2}


def _random_matrix(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    """Random numeric frame standing in for post-preprocessing features."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_rows, n_cols)).astype(np.float32),
        columns=[f"f{i}" for i in range(n_cols)],
    )


def test_mlp_classification_fit_predict() -> None:
    """The MLP trains a couple of epochs and honours the classifier contract."""
    X = _random_matrix(160, 5, seed=21)
    y = (X["f0"].to_numpy() + X["f1"].to_numpy() > 0).astype(np.int64)

    model = MODEL_REGISTRY.get("mlp")(MLP_PARAMS, TaskType.CLASSIFICATION, seed=0)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)) <= {0, 1}

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_mlp_regression_fit_predict() -> None:
    """The MLP handles regression: squeezed float predictions, no proba."""
    X = _random_matrix(160, 5, seed=22)
    y = (2.0 * X["f0"].to_numpy() - X["f2"].to_numpy()).astype(np.float64)

    model = MODEL_REGISTRY.get("mlp")(MLP_PARAMS, TaskType.REGRESSION, seed=0)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.issubdtype(preds.dtype, np.floating)
    assert np.all(np.isfinite(preds))
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)
