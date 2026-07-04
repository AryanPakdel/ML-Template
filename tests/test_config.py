"""Config loading: real experiment YAML, overrides, and cross-field validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ml_pipeline.config.loader import load_config
from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.core.types import TaskType

if TYPE_CHECKING:
    from tests.conftest import ConfigFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TITANIC = REPO_ROOT / "configs" / "experiment" / "smoke_titanic.yaml"


def test_load_smoke_titanic() -> None:
    """The real smoke experiment resolves groups into a valid PipelineConfig."""
    cfg = load_config(SMOKE_TITANIC)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.data.target == "Survived"
    assert cfg.data.task == TaskType.CLASSIFICATION
    assert cfg.model.name == "random_forest"
    assert cfg.training.cv.n_splits == 2
    assert cfg.training.cv.strategy == "stratified"


def test_set_override_changes_value() -> None:
    """--set style dot-path overrides win over group files."""
    cfg = load_config(SMOKE_TITANIC, overrides=["training.cv.n_splits=4", "run.seed=7"])
    assert cfg.training.cv.n_splits == 4
    assert cfg.run.seed == 7


def test_unknown_key_raises() -> None:
    """extra='forbid' rejects typo'd keys instead of silently ignoring them."""
    with pytest.raises(ValidationError):
        load_config(SMOKE_TITANIC, overrides=["data.bogus_key=1"])


def test_regression_with_stratified_split_raises(
    make_config: ConfigFactory,
    reg_csv_path: Path,
    reg_columns: list[dict],
) -> None:
    """Stratified splitting is classification-only."""
    with pytest.raises(ValidationError, match="stratified"):
        make_config(
            reg_csv_path,
            reg_columns,
            "value",
            "regression",
            split={"strategy": "stratified"},
        )


def test_time_split_without_time_column_raises(
    make_config: ConfigFactory,
    reg_csv_path: Path,
    reg_columns: list[dict],
) -> None:
    """strategy='time' requires split.time_column."""
    with pytest.raises(ValidationError, match="time_column"):
        make_config(
            reg_csv_path,
            reg_columns,
            "value",
            "regression",
            split={"strategy": "time"},
        )
