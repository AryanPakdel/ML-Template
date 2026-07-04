"""The FastAPI serving layer, driven entirely by bundle metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.serving.app import create_app

VALID_RECORD = {"num_a": 5.0, "num_b": 0.3, "cat_a": "mid"}


@pytest.fixture(scope="module")
def client(
    trained_bundle: tuple[Path, PipelineConfig],
    tmp_path_factory: pytest.TempPathFactory,
) -> TestClient:
    """TestClient over an app built from the session-trained bundle."""
    bundle_path, _ = trained_bundle
    artifacts_dir = tmp_path_factory.mktemp("serving_artifacts")
    return TestClient(create_app(bundle_path, artifacts_dir=str(artifacts_dir)))


def test_health(client: TestClient) -> None:
    """Liveness probe reports ok and names the loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "logistic_regression"


def test_model_info(client: TestClient) -> None:
    """Model info exposes identity and the raw feature columns."""
    response = client.get("/model_info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "logistic_regression"
    assert body["feature_columns"] == ["num_a", "num_b", "cat_a"]
    assert body["task"] == "classification"


def test_predict_valid(client: TestClient) -> None:
    """A valid record predicts one of the known class labels with probabilities."""
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert set(body["probabilities"]) == {"0", "1"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_wrong_type_422(client: TestClient) -> None:
    """A non-numeric value in a float field fails pydantic validation."""
    response = client.post("/predict", json={**VALID_RECORD, "num_a": "not-a-number"})
    assert response.status_code == 422


def test_predict_out_of_range_400(client: TestClient) -> None:
    """Values violating the trained schema's bounds are rejected with details."""
    response = client.post("/predict", json={**VALID_RECORD, "num_a": 50.0})
    assert response.status_code == 400
    assert any("num_a" in violation for violation in response.json()["detail"])


def test_predict_bad_category_400(client: TestClient) -> None:
    """Values outside allowed_values are rejected with details."""
    response = client.post("/predict", json={**VALID_RECORD, "cat_a": "purple"})
    assert response.status_code == 400
    assert any("cat_a" in violation for violation in response.json()["detail"])


def test_predict_batch(client: TestClient) -> None:
    """Batch endpoint returns one prediction per record, in order."""
    records = [VALID_RECORD, {"num_a": 1.0, "num_b": -1.2, "cat_a": "high"}]
    response = client.post("/predict_batch", json={"records": records})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert all(p in (0, 1) for p in body["predictions"])
    assert len(body["probabilities"]) == 2
