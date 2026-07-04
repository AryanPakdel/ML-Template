"""Missing-value analyzer: per-column null counts and percentages."""

from __future__ import annotations

import logging

import pandas as pd

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.eda.base import EDA_REGISTRY, EdaAnalyzer, EdaSection, df_to_html_table

logger = logging.getLogger(__name__)

_HIGH_MISSING_PCT: float = 20.0
"""Null percentage above which a column earns a finding (no EdaConfig knob exists)."""


@EDA_REGISTRY.register("missing")
class MissingAnalyzer(EdaAnalyzer):
    """Tabulate null count and share for every column that has any nulls."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the missing-values section; notes explicitly when there are none."""
        title = "Missing values"
        n_rows = len(df)
        null_counts = df.isna().sum()
        with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)

        if with_nulls.empty:
            return EdaSection(title, "<p>No missing values detected in any column.</p>")

        rows: list[dict[str, object]] = []
        findings: list[str] = []
        for column, count in with_nulls.items():
            pct = 100.0 * int(count) / n_rows if n_rows else 0.0
            rows.append(
                {"column": column, "null_count": int(count), "null_pct": f"{pct:.1f}%"}
            )
            if pct > _HIGH_MISSING_PCT:
                findings.append(
                    f"Column '{column}' is {pct:.1f}% missing ({int(count)}/{n_rows} rows) — "
                    "consider dropping it or configuring a dedicated imputer."
                )

        intro = (
            f"<p>{len(with_nulls)} of {df.shape[1]} columns contain missing values "
            f"({n_rows} rows total).</p>"
        )
        html = intro + df_to_html_table(pd.DataFrame(rows))
        return EdaSection(title, html, findings)
