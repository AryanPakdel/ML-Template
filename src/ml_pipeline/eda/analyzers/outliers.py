"""Outlier analyzer: IQR and/or z-score outlier counts per numeric column."""

from __future__ import annotations

import logging
import math

import pandas as pd

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.eda.base import (
    EDA_REGISTRY,
    EdaAnalyzer,
    EdaSection,
    df_to_html_table,
    numeric_feature_names,
)

logger = logging.getLogger(__name__)

_HIGH_OUTLIER_PCT: float = 5.0
"""Outlier percentage above which a column earns a finding."""


def _iqr_outlier_count(series: pd.Series, factor: float) -> int:
    """Count values outside ``[q1 - factor*iqr, q3 + factor*iqr]``."""
    if series.empty:
        return 0
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return int(((series < lower) | (series > upper)).sum())


def _zscore_outlier_count(series: pd.Series, threshold: float) -> int:
    """Count values whose |z-score| exceeds ``threshold`` (0 for constant columns)."""
    if len(series) < 2:
        return 0
    std = float(series.std())
    if std == 0.0 or math.isnan(std):
        return 0
    z_scores = (series - float(series.mean())) / std
    return int((z_scores.abs() > threshold).sum())


@EDA_REGISTRY.register("outliers")
class OutliersAnalyzer(EdaAnalyzer):
    """Count IQR/z-score outliers per numeric feature column per config method."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the outlier section using ``eda_cfg.outlier_method``."""
        title = "Outliers"
        numeric_cols = numeric_feature_names(df, data_cfg)
        if not numeric_cols:
            return EdaSection(title, "<p>No numeric feature columns found.</p>")

        method = eda_cfg.outlier_method
        use_iqr = method in ("iqr", "both")
        use_zscore = method in ("zscore", "both")

        rows: list[dict[str, object]] = []
        findings: list[str] = []
        for column in numeric_cols:
            series = df[column].dropna().astype(float)
            n_non_null = len(series)
            row: dict[str, object] = {"column": column, "non_null": n_non_null}
            worst_pct = 0.0
            if use_iqr:
                count = _iqr_outlier_count(series, eda_cfg.iqr_factor)
                pct = 100.0 * count / n_non_null if n_non_null else 0.0
                row["iqr_outliers"] = count
                row["iqr_pct"] = f"{pct:.1f}%"
                worst_pct = max(worst_pct, pct)
            if use_zscore:
                count = _zscore_outlier_count(series, eda_cfg.zscore_threshold)
                pct = 100.0 * count / n_non_null if n_non_null else 0.0
                row["zscore_outliers"] = count
                row["zscore_pct"] = f"{pct:.1f}%"
                worst_pct = max(worst_pct, pct)
            rows.append(row)
            if worst_pct > _HIGH_OUTLIER_PCT:
                findings.append(
                    f"Column '{column}' has {worst_pct:.1f}% outliers (method='{method}') — "
                    "consider preprocessing.numeric.outliers (clip/remove)."
                )

        intro = (
            f"<p>Method: <code>{method}</code> "
            f"(iqr_factor={eda_cfg.iqr_factor:g}, "
            f"zscore_threshold={eda_cfg.zscore_threshold:g}).</p>"
        )
        html = intro + df_to_html_table(pd.DataFrame(rows))
        return EdaSection(title, html, findings)
