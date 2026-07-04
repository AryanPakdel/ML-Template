"""Correlation analyzer: numeric heatmap plus ranked |corr| with the target."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
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
    target_series_numeric,
)

logger = logging.getLogger(__name__)

_ANNOTATE_MAX_COLS: int = 15
"""Annotate heatmap cells with values only up to this matrix size."""


@EDA_REGISTRY.register("correlation")
class CorrelationAnalyzer(EdaAnalyzer):
    """Heatmap of numeric correlations and a ranked table of target correlations."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the correlation section; skipped when <2 numeric columns exist."""
        title = "Correlations"
        numeric_cols = numeric_feature_names(df, data_cfg)
        target_numeric = target_series_numeric(df, data_cfg)

        n_matrix_cols = len(numeric_cols) + (0 if target_numeric is None else 1)
        if n_matrix_cols < 2:
            logger.info("correlation: fewer than 2 numeric columns; section skipped.")
            return None

        matrix = df[numeric_cols].astype(float)
        if target_numeric is not None:
            matrix = matrix.assign(**{data_cfg.target: target_numeric})
        corr = matrix.corr()

        side = max(6.0, 0.6 * len(corr))
        fig, ax = plt.subplots(figsize=(side, side * 0.85))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            annot=len(corr) <= _ANNOTATE_MAX_COLS,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation matrix (numeric columns; categorical target factorized)")
        html_parts = [fig_to_html(fig)]

        findings: list[str] = []
        if target_numeric is not None and numeric_cols:
            with_target = df[numeric_cols].astype(float).corrwith(target_numeric).dropna()
            ranked = with_target.reindex(with_target.abs().sort_values(ascending=False).index)
            rows = [
                {
                    "feature": feature,
                    "corr_with_target": round(float(value), 4),
                    "abs_corr": round(abs(float(value)), 4),
                }
                for feature, value in ranked.items()
            ]
            if rows:
                html_parts.append(
                    f"<h3>Correlation with target '{data_cfg.target}'</h3>"
                    + df_to_html_table(pd.DataFrame(rows))
                )
                top_feature = ranked.index[0]
                findings.append(
                    f"Strongest linear association with target '{data_cfg.target}': "
                    f"'{top_feature}' (corr={float(ranked.iloc[0]):.3f})."
                )
        else:
            html_parts.append(
                "<p>Target correlations unavailable (target missing or no numeric "
                "features).</p>"
            )

        return EdaSection(title, "".join(html_parts), findings)
