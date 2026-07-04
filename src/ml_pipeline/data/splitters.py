"""Train/val/test splitters: config strategy key -> :class:`DatasetSplits`.

The two-stage random/stratified splits adjust the validation fraction to
``val_size / (1 - test_size)`` so the final proportions match the config
exactly; the time splitter never shuffles and keeps temporal order.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pipeline.config.schema import DataConfig
from ml_pipeline.core.registry import Registry
from ml_pipeline.core.types import DatasetSplits

logger = logging.getLogger(__name__)

SPLITTER_REGISTRY: Registry[type["BaseSplitter"]] = Registry("splitter")


class BaseSplitter(ABC):
    """Contract for splitters: validated frame in, three partitions out."""

    @abstractmethod
    def split(self, df: pd.DataFrame, data_cfg: DataConfig, seed: int) -> DatasetSplits:
        """Partition ``df`` into train/val/test per ``data_cfg.split``."""


def _two_stage_split(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    seed: int,
    stratify_column: str | None,
) -> DatasetSplits:
    """Shared random/stratified logic: peel off test, then split train/val.

    Args:
        df: full validated dataset.
        data_cfg: data config providing split sizes.
        seed: random seed for both stages.
        stratify_column: column to stratify on, or ``None`` for plain random.

    Returns:
        The three partitions with config-accurate proportions.
    """
    split_cfg = data_cfg.split
    stratify_full = df[stratify_column] if stratify_column else None
    train_val, test = train_test_split(
        df,
        test_size=split_cfg.test_size,
        random_state=seed,
        stratify=stratify_full,
    )
    # Rescale so val ends up as val_size of the *original* dataset.
    val_fraction = split_cfg.val_size / (1.0 - split_cfg.test_size)
    stratify_rest = train_val[stratify_column] if stratify_column else None
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_rest,
    )
    return DatasetSplits(train=train, val=val, test=test)


@SPLITTER_REGISTRY.register("random")
class RandomSplitter(BaseSplitter):
    """Uniform random two-stage split."""

    def split(self, df: pd.DataFrame, data_cfg: DataConfig, seed: int) -> DatasetSplits:
        """Randomly partition rows into train/val/test."""
        return _two_stage_split(df, data_cfg, seed, stratify_column=None)


@SPLITTER_REGISTRY.register("stratified")
class StratifiedSplitter(BaseSplitter):
    """Random split preserving class proportions in every partition."""

    def split(self, df: pd.DataFrame, data_cfg: DataConfig, seed: int) -> DatasetSplits:
        """Stratify both stages on ``split.stratify_column`` (default: the target)."""
        column = data_cfg.split.stratify_column or data_cfg.target
        if column not in df.columns:
            raise KeyError(f"Stratify column '{column}' not found in dataset columns.")
        return _two_stage_split(df, data_cfg, seed, stratify_column=column)


@SPLITTER_REGISTRY.register("time")
class TimeSplitter(BaseSplitter):
    """Chronological split: oldest rows train, newest rows test. No shuffling."""

    def split(self, df: pd.DataFrame, data_cfg: DataConfig, seed: int) -> DatasetSplits:
        """Sort by ``split.time_column`` ascending and slice sequentially.

        ``seed`` is accepted for interface parity but never used — temporal
        splits must be deterministic and order-preserving.
        """
        split_cfg = data_cfg.split
        time_column = split_cfg.time_column
        if time_column is None:  # enforced by config; defensive here
            raise ValueError("split.strategy='time' requires split.time_column")
        if time_column not in df.columns:
            raise KeyError(f"Time column '{time_column}' not found in dataset columns.")

        ordered = df.sort_values(time_column, ascending=True, kind="stable")
        n = len(ordered)
        n_test = int(round(n * split_cfg.test_size))
        n_val = int(round(n * split_cfg.val_size))
        n_train = n - n_val - n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"Time split produced an empty partition "
                f"(train={n_train}, val={n_val}, test={n_test}) from {n} rows."
            )
        return DatasetSplits(
            train=ordered.iloc[:n_train],
            val=ordered.iloc[n_train : n_train + n_val],
            test=ordered.iloc[n_train + n_val :],
        )


def split_dataset(df: pd.DataFrame, data_cfg: DataConfig, seed: int) -> DatasetSplits:
    """Split the dataset by dispatching the configured strategy through the registry.

    Args:
        df: validated dataset (all original columns, target included).
        data_cfg: data config carrying the split strategy and sizes.
        seed: run-level random seed.

    Returns:
        Train/val/test partitions.

    Raises:
        KeyError: if the strategy key is not registered.
    """
    splitter_cls = SPLITTER_REGISTRY.get(data_cfg.split.strategy)
    splits = splitter_cls().split(df, data_cfg, seed)
    logger.info(
        "Split dataset with '%s' strategy: %s", data_cfg.split.strategy, splits.sizes()
    )
    return splits
