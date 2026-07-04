"""Built-in EDA analyzers; importing this package registers them all.

Each module defines one :class:`~ml_pipeline.eda.base.EdaAnalyzer` subclass and
registers it in :data:`~ml_pipeline.eda.base.EDA_REGISTRY` at import time.
"""

from __future__ import annotations

from ml_pipeline.eda.analyzers import (
    cardinality,
    categoricals,
    class_balance,
    correlation,
    distributions,
    leakage,
    missing,
    outliers,
)

__all__ = [
    "cardinality",
    "categoricals",
    "class_balance",
    "correlation",
    "distributions",
    "leakage",
    "missing",
    "outliers",
]
