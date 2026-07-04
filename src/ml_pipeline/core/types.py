"""Shared enums and data contracts used across pipeline stages.

Stages never import each other's internals; they communicate through these types,
validated config objects, and persisted artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class TaskType(StrEnum):
    """Supported supervised learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ExplainerHint(StrEnum):
    """Which SHAP explainer family suits a model; ``NONE`` skips explainability."""

    TREE = "tree"
    LINEAR = "linear"
    KERNEL = "kernel"
    NONE = "none"


class ColumnRole(StrEnum):
    """How a raw column participates in the pipeline.

    - ``FEATURE``: used as a model input.
    - ``ID``: kept for reporting/error analysis, excluded from model inputs.
    - ``DROP``: removed immediately after validation.
    """

    FEATURE = "feature"
    ID = "id"
    DROP = "drop"


@dataclass(frozen=True)
class DatasetSplits:
    """Raw train/validation/test partitions produced by a splitter.

    All three frames keep the original (validated) columns, including the target.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        """Row counts per split, for logging."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}
