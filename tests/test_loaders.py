"""Dataset loaders: registry dispatch, round-trips, and error paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_pipeline.config.schema import DataSourceConfig
from ml_pipeline.data.loaders import load_dataframe


@pytest.fixture()
def small_frame() -> pd.DataFrame:
    """A tiny frame with mixed dtypes for round-trip checks."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5], "c": ["x", "y", "z"]})


def test_csv_round_trip(tmp_path: Path, small_frame: pd.DataFrame) -> None:
    """CSV written to disk loads back identically through the registry."""
    path = tmp_path / "data.csv"
    small_frame.to_csv(path, index=False)
    loaded = load_dataframe(DataSourceConfig(type="csv", path=str(path)))
    pd.testing.assert_frame_equal(loaded, small_frame)


def test_parquet_round_trip(tmp_path: Path, small_frame: pd.DataFrame) -> None:
    """Parquet written to disk loads back identically through the registry."""
    path = tmp_path / "data.parquet"
    small_frame.to_parquet(path, index=False)
    loaded = load_dataframe(DataSourceConfig(type="parquet", path=str(path)))
    pd.testing.assert_frame_equal(loaded, small_frame)


def test_unknown_loader_type_raises(tmp_path: Path) -> None:
    """An unregistered loader key is a KeyError listing valid loaders."""
    cfg = DataSourceConfig(type="sqlite", path=str(tmp_path / "db.sqlite"))
    with pytest.raises(KeyError, match="loader"):
        load_dataframe(cfg)


def test_missing_file_raises(tmp_path: Path) -> None:
    """A resolvable loader with a nonexistent path raises FileNotFoundError."""
    cfg = DataSourceConfig(type="csv", path=str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_dataframe(cfg)
