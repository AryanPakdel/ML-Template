"""Data layer: loading, schema validation, and train/val/test splitting.

Importing this package populates ``LOADER_REGISTRY`` and ``SPLITTER_REGISTRY``
with the built-in implementations; add new ones by registering in their module
and importing it here.
"""

from __future__ import annotations

from ml_pipeline.data import loaders, splitters, validation  # noqa: F401
from ml_pipeline.data.loaders import LOADER_REGISTRY, BaseLoader, load_dataframe
from ml_pipeline.data.splitters import SPLITTER_REGISTRY, BaseSplitter, split_dataset
from ml_pipeline.data.validation import (
    DataValidationError,
    build_pandera_schema,
    validate_dataframe,
)

__all__ = [
    "LOADER_REGISTRY",
    "SPLITTER_REGISTRY",
    "BaseLoader",
    "BaseSplitter",
    "DataValidationError",
    "build_pandera_schema",
    "load_dataframe",
    "split_dataset",
    "validate_dataframe",
]
