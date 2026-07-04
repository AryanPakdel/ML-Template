"""Classical models via the registry: fit/predict/predict_proba contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.core.types import TaskType
from ml_pipeline.models import MODEL_REGISTRY
from ml_pipeline.preprocessing.builder import build_preprocessor

#: Small, fast hyperparameters per tested classifier.
_CLF_PARAMS: dict[str, dict] = {
    "logistic_regression": {"max_iter": 500},
    "random_forest": {"n_estimators": 25},
    "xgboost": {"n_estimators": 25},
    "lightgbm": {"n_estimators": 25},
}


@pytest.fixture(scope="module")
def clf_matrix(
    synthetic_classification_df: pd.DataFrame, clf_config: PipelineConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Post-preprocessing classification features and encoded target."""
    X = synthetic_classification_df[["num_a", "num_b", "cat_a"]]
    preprocessor = build_preprocessor(clf_config.preprocessing, clf_config.data, train_df=X)
    return preprocessor.fit_transform(X), synthetic_classification_df["label"].to_numpy()


@pytest.fixture(scope="module")
def reg_matrix(
    synthetic_regression_df: pd.DataFrame, reg_config: PipelineConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Post-preprocessing regression features and continuous target."""
    X = synthetic_regression_df[["num_a", "num_b", "cat_a"]]
    preprocessor = build_preprocessor(reg_config.preprocessing, reg_config.data, train_df=X)
    return preprocessor.fit_transform(X), synthetic_regression_df["value"].to_numpy()


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
)
def test_classifier_contract(model_name: str, clf_matrix: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Every classifier fits and emits well-shaped predictions/probabilities."""
    X, y = clf_matrix
    model_cls = MODEL_REGISTRY.get(model_name)
    model = model_cls(_CLF_PARAMS[model_name], TaskType.CLASSIFICATION, seed=0)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)) <= {0, 1}

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_linear_regression_contract(reg_matrix: tuple[pd.DataFrame, np.ndarray]) -> None:
    """OLS fits the linear synthetic data and predicts with the right shape."""
    X, y = reg_matrix
    model_cls = MODEL_REGISTRY.get("linear_regression")
    model = model_cls({}, TaskType.REGRESSION, seed=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))
    assert r2_score(y, preds) > 0.9  # target is (nearly) linear in the features
