"""Cardinality analyzer: distinct-value counts for categorical/string columns."""

from __future__ import annotations

import logging

import pandas as pd

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.eda.base import (
    EDA_REGISTRY,
    EdaAnalyzer,
    EdaSection,
    categorical_feature_names,
    df_to_html_table,
)

logger = logging.getLogger(__name__)


@EDA_REGISTRY.register("cardinality")
class CardinalityAnalyzer(EdaAnalyzer):
    """Report ``nunique`` per categorical/string feature column and flag high values."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the cardinality section using ``eda_cfg.high_cardinality_threshold``."""
        title = "Cardinality"
        columns = categorical_feature_names(df, data_cfg)
        if not columns:
            return EdaSection(title, "<p>No categorical/string feature columns found.</p>")

        rows: list[dict[str, object]] = []
        findings: list[str] = []
        for column in columns:
            non_null = int(df[column].notna().sum())
            distinct = int(df[column].nunique(dropna=True))
            unique_pct = 100.0 * distinct / non_null if non_null else 0.0
            rows.append(
                {
                    "column": column,
                    "distinct_values": distinct,
                    "non_null": non_null,
                    "unique_pct": f"{unique_pct:.1f}%",
                }
            )
            if distinct > eda_cfg.high_cardinality_threshold:
                findings.append(
                    f"Column '{column}' has {distinct} distinct values "
                    f"(> {eda_cfg.high_cardinality_threshold}) — one-hot encoding would "
                    "explode; prefer frequency/target encoding."
                )

        rows.sort(
            key=lambda row: int(row["distinct_values"]),  # type: ignore[arg-type]
            reverse=True,
        )
        intro = (
            f"<p>High-cardinality threshold: "
            f"{eda_cfg.high_cardinality_threshold} distinct values.</p>"
        )
        html = intro + df_to_html_table(pd.DataFrame(rows))
        return EdaSection(title, html, findings)
