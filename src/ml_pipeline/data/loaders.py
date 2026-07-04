"""Dataset loaders: turn a :class:`DataSourceConfig` into a raw ``pd.DataFrame``.

New formats plug in by subclassing :class:`BaseLoader` and registering under a
string key; configs then select them via ``data.source.type`` without touching
orchestration code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ml_pipeline.config.schema import DataSourceConfig
from ml_pipeline.core.registry import Registry

logger = logging.getLogger(__name__)

LOADER_REGISTRY: Registry[type["BaseLoader"]] = Registry("loader")


class BaseLoader(ABC):
    """Contract for raw-data loaders: one config in, one DataFrame out."""

    @abstractmethod
    def load(self, cfg: DataSourceConfig) -> pd.DataFrame:
        """Read the dataset described by ``cfg`` into a DataFrame."""

    def _resolve_path(self, cfg: DataSourceConfig) -> Path:
        """Resolve ``cfg.path`` relative to the current working directory.

        Raises:
            FileNotFoundError: with a hint to fetch the demo datasets first.
        """
        path = Path(cfg.path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. "
                "If this is a demo dataset, run "
                "'python scripts/download_data.py --dataset all' first."
            )
        return path


@LOADER_REGISTRY.register("csv")
class CsvLoader(BaseLoader):
    """Load a CSV file with ``pandas.read_csv``; ``read_options`` pass through."""

    def load(self, cfg: DataSourceConfig) -> pd.DataFrame:
        """Read the CSV at ``cfg.path`` applying ``cfg.read_options``."""
        path = self._resolve_path(cfg)
        return pd.read_csv(path, **cfg.read_options)


@LOADER_REGISTRY.register("parquet")
class ParquetLoader(BaseLoader):
    """Load a Parquet file with ``pandas.read_parquet``; ``read_options`` pass through."""

    def load(self, cfg: DataSourceConfig) -> pd.DataFrame:
        """Read the Parquet file at ``cfg.path`` applying ``cfg.read_options``."""
        path = self._resolve_path(cfg)
        return pd.read_parquet(path, **cfg.read_options)


def load_dataframe(cfg: DataSourceConfig) -> pd.DataFrame:
    """Load the raw dataset by dispatching ``cfg.type`` through the loader registry.

    Args:
        cfg: source section of the data config (loader key, path, read options).

    Returns:
        The raw, unvalidated DataFrame.

    Raises:
        KeyError: if ``cfg.type`` is not a registered loader key.
        FileNotFoundError: if the resolved path does not exist.
    """
    loader_cls = LOADER_REGISTRY.get(cfg.type)
    df = loader_cls().load(cfg)
    logger.info(
        "Loaded dataset via '%s' loader from %s: %d rows x %d columns",
        cfg.type,
        cfg.path,
        df.shape[0],
        df.shape[1],
    )
    return df
