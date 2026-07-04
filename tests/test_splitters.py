"""Train/val/test splitters: sizes, stratification, and temporal ordering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.data.splitters import split_dataset

if TYPE_CHECKING:
    from tests.conftest import ConfigFactory


def test_random_split_sizes_match_config(
    synthetic_classification_df: pd.DataFrame,
    make_config: ConfigFactory,
    clf_csv_path: Path,
    clf_columns: list[dict],
) -> None:
    """Random split proportions land within rounding of the configured sizes."""
    cfg = make_config(clf_csv_path, clf_columns, "label", "classification")
    df = synthetic_classification_df
    splits = split_dataset(df, cfg.data, seed=cfg.run.seed)
    n = len(df)
    assert len(splits.train) + len(splits.val) + len(splits.test) == n
    assert len(splits.test) == pytest.approx(cfg.data.split.test_size * n, abs=2)
    assert len(splits.val) == pytest.approx(cfg.data.split.val_size * n, abs=2)
    # Partitions are disjoint on the original row index.
    combined = np.concatenate([splits.train.index, splits.val.index, splits.test.index])
    assert len(set(combined)) == n


def test_stratified_split_preserves_class_ratio(
    synthetic_classification_df: pd.DataFrame, clf_config: PipelineConfig
) -> None:
    """Every partition keeps the overall positive-class ratio within 5 points."""
    df = synthetic_classification_df
    overall = df["label"].mean()
    splits = split_dataset(df, clf_config.data, seed=clf_config.run.seed)
    for part in (splits.train, splits.val, splits.test):
        assert part["label"].mean() == pytest.approx(overall, abs=0.05)


def test_time_split_preserves_order(make_config: ConfigFactory, tmp_path: Path) -> None:
    """Chronological split: train times <= val times <= test times."""
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "t": rng.permutation(n),  # shuffled on purpose; splitter must sort
            "num_a": rng.uniform(0.5, 9.5, n),
            "value": rng.normal(0.0, 1.0, n),
        }
    )
    csv_path = tmp_path / "timed.csv"
    df.to_csv(csv_path, index=False)
    cfg = make_config(
        csv_path,
        [
            {"name": "t", "dtype": "int"},
            {"name": "num_a", "dtype": "float"},
            {"name": "value", "dtype": "float"},
        ],
        "value",
        "regression",
        split={"strategy": "time", "time_column": "t"},
        training={"cv": {"strategy": "timeseries", "n_splits": 2, "shuffle": False}},
    )
    splits = split_dataset(df, cfg.data, seed=cfg.run.seed)
    assert splits.train["t"].max() <= splits.val["t"].min()
    assert splits.val["t"].max() <= splits.test["t"].min()
    assert len(splits.train) + len(splits.val) + len(splits.test) == n
