"""The preprocessing builder: dense pandas output and the cardinality guard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.preprocessing.builder import build_preprocessor

if TYPE_CHECKING:
    from tests.conftest import ConfigFactory


def test_fit_transform_dense_pandas_no_nan(
    synthetic_classification_df: pd.DataFrame, clf_config: PipelineConfig
) -> None:
    """Default preprocessing yields an all-numeric, NaN-free pandas DataFrame."""
    X = synthetic_classification_df[["num_a", "num_b", "cat_a"]].copy()
    # Inject missing values: imputers must handle them inside the pipeline.
    X.loc[X.index[:5], "num_a"] = np.nan
    X.loc[X.index[5:10], "cat_a"] = None

    preprocessor = build_preprocessor(clf_config.preprocessing, clf_config.data, train_df=X)
    transformed = preprocessor.fit_transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == len(X)
    assert not transformed.isna().any().any()
    expected_onehot = {"cat_a_low", "cat_a_mid", "cat_a_high"}
    assert expected_onehot <= set(transformed.columns)
    assert {"num_a", "num_b"} <= set(transformed.columns)


def test_high_cardinality_degrades_to_frequency(
    make_config: ConfigFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Above max_onehot_cardinality, one-hot degrades to one frequency column."""
    rng = np.random.default_rng(11)
    n = 120
    df = pd.DataFrame(
        {
            "num_a": rng.uniform(0.5, 9.5, n),
            "cat_hc": rng.choice([f"c{i}" for i in range(30)], size=n),
            "label": rng.integers(0, 2, n),
        }
    )
    csv_path = tmp_path / "hc.csv"
    df.to_csv(csv_path, index=False)
    cfg = make_config(
        csv_path,
        [
            {"name": "num_a", "dtype": "float"},
            {"name": "cat_hc", "dtype": "category"},
            {"name": "label", "dtype": "int"},
        ],
        "label",
        "classification",
        overrides={"preprocessing": {"categorical": {"max_onehot_cardinality": 5}}},
    )

    X = df[["num_a", "cat_hc"]]
    with caplog.at_level(logging.WARNING, logger="ml_pipeline.preprocessing.builder"):
        preprocessor = build_preprocessor(cfg.preprocessing, cfg.data, train_df=X)
    assert any("frequency" in record.message for record in caplog.records)

    transformed = preprocessor.fit_transform(X)
    assert "cat_hc" in transformed.columns  # single frequency-encoded column
    assert not any(c.startswith("cat_hc_") for c in transformed.columns)
    assert transformed.shape[1] == 2  # num_a + cat_hc, no one-hot explosion
