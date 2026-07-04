"""Categorical analyzer: bar plots of top value counts per categorical column."""

from __future__ import annotations

import logging

import pandas as pd

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.eda.base import (
    EDA_REGISTRY,
    EdaAnalyzer,
    EdaSection,
    categorical_feature_names,
    fig_to_html,
    subplot_grid,
)

logger = logging.getLogger(__name__)

_TOP_VALUES: int = 20
"""Number of most frequent values plotted per categorical column."""


@EDA_REGISTRY.register("categoricals")
class CategoricalsAnalyzer(EdaAnalyzer):
    """Bar-plot the top value counts of each categorical/string feature column."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the categorical value-count section."""
        title = "Categorical value counts"
        columns = categorical_feature_names(df, data_cfg)
        if not columns:
            return EdaSection(title, "<p>No categorical/string feature columns found.</p>")

        fig, axes = subplot_grid(len(columns))
        truncated: list[str] = []
        for ax, column in zip(axes, columns, strict=True):
            counts = df[column].value_counts(dropna=True).head(_TOP_VALUES)
            if counts.empty:
                ax.set_title(f"{column} (no values)")
                continue
            labels = [str(value) for value in counts.index]
            ax.bar(range(len(counts)), counts.to_numpy())
            ax.set_xticks(range(len(counts)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(column)
            distinct = int(df[column].nunique(dropna=True))
            if distinct > _TOP_VALUES:
                truncated.append(f"'{column}' (top {_TOP_VALUES} of {distinct} values shown)")
        fig.tight_layout()

        note = ""
        if truncated:
            note = f"<p><em>Truncated columns: {', '.join(truncated)}.</em></p>"
        html = note + fig_to_html(fig)
        return EdaSection(title, html)
