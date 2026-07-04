"""Distribution analyzer: histogram + KDE and skewness for numeric features."""

from __future__ import annotations

import logging
import math

import pandas as pd
import seaborn as sns

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.eda.base import (
    EDA_REGISTRY,
    EdaAnalyzer,
    EdaSection,
    df_to_html_table,
    fig_to_html,
    numeric_feature_names,
    subplot_grid,
)

logger = logging.getLogger(__name__)

_HIGH_SKEW: float = 1.0
"""|skewness| above which a column is flagged as heavily skewed."""


@EDA_REGISTRY.register("distributions")
class DistributionsAnalyzer(EdaAnalyzer):
    """Plot histograms (with KDE) and report skewness for numeric feature columns."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the distributions section, capped at ``eda_cfg.max_distribution_plots``."""
        title = "Numeric distributions"
        numeric_cols = numeric_feature_names(df, data_cfg)
        if not numeric_cols:
            return EdaSection(title, "<p>No numeric feature columns found.</p>")

        plotted = numeric_cols[: eda_cfg.max_distribution_plots]
        if len(plotted) < len(numeric_cols):
            logger.info(
                "distributions: plotting %d of %d numeric columns (max_distribution_plots).",
                len(plotted),
                len(numeric_cols),
            )

        fig, axes = subplot_grid(len(plotted))
        rows: list[dict[str, object]] = []
        findings: list[str] = []
        for ax, column in zip(axes, plotted, strict=True):
            series = df[column].dropna().astype(float)
            skew = float(series.skew()) if len(series) > 2 else math.nan
            rows.append(
                {"column": column, "skewness": None if math.isnan(skew) else round(skew, 3)}
            )
            if not math.isnan(skew) and abs(skew) > _HIGH_SKEW:
                findings.append(
                    f"Column '{column}' is heavily skewed (skewness={skew:.2f}) — "
                    "a log/power transform may help linear models."
                )
            if series.empty:
                ax.set_title(f"{column} (all null)")
                continue
            sns.histplot(x=series, kde=series.nunique() > 1, ax=ax)
            ax.set_title(column)
            ax.set_xlabel("")
        fig.tight_layout()

        note = ""
        if len(plotted) < len(numeric_cols):
            note = (
                f"<p><em>Showing first {len(plotted)} of {len(numeric_cols)} numeric "
                "columns (eda.max_distribution_plots).</em></p>"
            )
        html = note + df_to_html_table(pd.DataFrame(rows)) + fig_to_html(fig)
        return EdaSection(title, html, findings)
