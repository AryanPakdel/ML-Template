"""Shared fixtures for the ml_pipeline test suite.

Everything lives under pytest-managed tmp directories: synthetic CSVs, run
artifacts, and logs. The suite never writes to the repository's ``artifacts/``,
``logs/``, or ``mlruns/`` directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.config.loader import deep_merge
from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.training.trainer import PipelineTrainer

N_ROWS = 240
SEED = 1234

#: ColumnSpec dicts for the synthetic classification dataset (target: "label").
CLF_COLUMNS: list[dict[str, Any]] = [
    {"name": "num_a", "dtype": "float", "ge": 0.0, "le": 10.0},
    {"name": "num_b", "dtype": "float"},
    {"name": "cat_a", "dtype": "category", "allowed_values": ["low", "mid", "high"]},
    {"name": "label", "dtype": "int", "allowed_values": [0, 1]},
]

#: ColumnSpec dicts for the synthetic regression dataset (target: "value").
REG_COLUMNS: list[dict[str, Any]] = [
    {"name": "num_a", "dtype": "float", "ge": 0.0, "le": 10.0},
    {"name": "num_b", "dtype": "float"},
    {"name": "cat_a", "dtype": "category", "allowed_values": ["low", "mid", "high"]},
    {"name": "value", "dtype": "float"},
]


class ConfigFactory(Protocol):
    """Callable signature of the :func:`make_config` fixture."""

    def __call__(
        self,
        csv_path: str | Path,
        columns: list[dict[str, Any]],
        target: str,
        task: str,
        *,
        split: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineConfig: ...


def _base_features(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shared synthetic feature columns: two floats and a 3-level category."""
    return {
        "num_a": rng.uniform(0.5, 9.5, N_ROWS),
        "num_b": rng.normal(0.0, 1.0, N_ROWS),
        "cat_a": rng.choice(["low", "mid", "high"], size=N_ROWS, p=[0.4, 0.35, 0.25]),
    }


@pytest.fixture(scope="session")
def synthetic_classification_df() -> pd.DataFrame:
    """~240-row seeded classification frame with a learnable binary target."""
    rng = np.random.default_rng(SEED)
    features = _base_features(rng)
    logits = (
        0.8 * (features["num_a"] - 5.0)
        + 1.2 * features["num_b"]
        + 1.5 * (features["cat_a"] == "high")
        + rng.normal(0.0, 1.0, N_ROWS)
    )
    features["label"] = (logits > 0).astype(np.int64)
    return pd.DataFrame(features)


@pytest.fixture(scope="session")
def synthetic_regression_df() -> pd.DataFrame:
    """~240-row seeded regression frame with a linear continuous target."""
    rng = np.random.default_rng(SEED + 1)
    features = _base_features(rng)
    features["value"] = (
        3.0 * features["num_a"]
        - 2.0 * features["num_b"]
        + 1.0 * (features["cat_a"] == "high")
        + rng.normal(0.0, 1.0, N_ROWS)
    )
    return pd.DataFrame(features)


@pytest.fixture(scope="session")
def clf_csv_path(
    synthetic_classification_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The classification frame persisted as a tmp CSV."""
    path = tmp_path_factory.mktemp("data") / "clf.csv"
    synthetic_classification_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def reg_csv_path(
    synthetic_regression_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The regression frame persisted as a tmp CSV."""
    path = tmp_path_factory.mktemp("data") / "reg.csv"
    synthetic_regression_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def clf_columns() -> list[dict[str, Any]]:
    """Fresh copies of the classification ColumnSpec dicts."""
    return [dict(spec) for spec in CLF_COLUMNS]


@pytest.fixture(scope="session")
def reg_columns() -> list[dict[str, Any]]:
    """Fresh copies of the regression ColumnSpec dicts."""
    return [dict(spec) for spec in REG_COLUMNS]


@pytest.fixture(scope="session")
def make_config(tmp_path_factory: pytest.TempPathFactory) -> ConfigFactory:
    """Factory building a validated :class:`PipelineConfig` rooted under tmp dirs.

    Defaults: mlflow disabled, 2-fold CV, explainability off, preprocessing and
    feature stages at their schema defaults. ``overrides`` deep-merges into the
    raw dict before validation.
    """

    def _make(
        csv_path: str | Path,
        columns: list[dict[str, Any]],
        target: str,
        task: str,
        *,
        split: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineConfig:
        run_root = tmp_path_factory.mktemp("run")
        raw: dict[str, Any] = {
            "run": {
                "experiment_name": "test",
                "seed": 42,
                "artifacts_dir": str(run_root / "artifacts"),
                "logs_dir": str(run_root / "logs"),
                "mlflow": {"enabled": False},
            },
            "data": {
                "source": {"type": "csv", "path": str(csv_path)},
                "target": target,
                "task": task,
                "columns": [dict(spec) for spec in columns],
                "split": split or {"strategy": "random"},
            },
            "training": training or {"cv": {"strategy": "kfold", "n_splits": 2}},
            "evaluation": {"explainability": {"method": "none"}},
        }
        if model is not None:
            raw["model"] = model
        if overrides:
            raw = deep_merge(raw, overrides)
        return PipelineConfig.model_validate(raw)

    return _make


@pytest.fixture(scope="session")
def clf_config(
    make_config: ConfigFactory, clf_csv_path: Path, clf_columns: list[dict[str, Any]]
) -> PipelineConfig:
    """Canonical classification config: stratified split, logistic regression."""
    return make_config(
        clf_csv_path,
        clf_columns,
        "label",
        "classification",
        split={"strategy": "stratified"},
        model={"name": "logistic_regression", "params": {"max_iter": 500}},
    )


@pytest.fixture(scope="session")
def reg_config(
    make_config: ConfigFactory, reg_csv_path: Path, reg_columns: list[dict[str, Any]]
) -> PipelineConfig:
    """Canonical regression config: random split, linear regression."""
    return make_config(
        reg_csv_path,
        reg_columns,
        "value",
        "regression",
        model={"name": "linear_regression"},
    )


@pytest.fixture(scope="session")
def trained_bundle(clf_config: PipelineConfig) -> tuple[Path, PipelineConfig]:
    """One real logistic-regression training run on the synthetic data.

    Returns:
        ``(bundle_path, cfg)`` — the persisted ``bundle.joblib`` (under a tmp
        artifacts dir) and the config that produced it.
    """
    result = PipelineTrainer(clf_config, run_cv=False, explain=False).run()
    return result.bundle_path, clf_config
