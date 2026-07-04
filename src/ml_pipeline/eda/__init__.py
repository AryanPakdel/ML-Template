"""Automated exploratory data analysis (EDA) stage.

Importing this package registers every built-in analyzer into ``EDA_REGISTRY``
(via the ``analyzers`` subpackage) and exposes :func:`run_eda`, which renders a
self-contained ``report.html`` plus a ``report.md`` findings summary.
"""

from __future__ import annotations

from ml_pipeline.eda import analyzers, base, report
from ml_pipeline.eda.base import EDA_REGISTRY, EdaAnalyzer, EdaSection, fig_to_html
from ml_pipeline.eda.report import run_eda

__all__ = [
    "EDA_REGISTRY",
    "EdaAnalyzer",
    "EdaSection",
    "analyzers",
    "base",
    "fig_to_html",
    "report",
    "run_eda",
]
