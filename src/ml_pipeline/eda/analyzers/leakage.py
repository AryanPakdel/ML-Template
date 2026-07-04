"""Leakage analyzer: near-perfect target correlations and duplicated target columns."""

from __future__ import annotations

import logging

import pandas as pd

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.eda.base import (
    EDA_REGISTRY,
    EdaAnalyzer,
    EdaSection,
    df_to_html_table,
    numeric_feature_names,
    target_series_numeric,
)

logger = logging.getLogger(__name__)


@EDA_REGISTRY.register("leakage")
class LeakageAnalyzer(EdaAnalyzer):
    """Flag features that correlate almost perfectly with — or duplicate — the target."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the leakage section; emits loud findings for suspicious columns."""
        title = "Target leakage checks"
        if data_cfg.target not in df.columns:
            logger.warning(
                "leakage: target '%s' not in DataFrame; section skipped.", data_cfg.target
            )
            return None

        threshold = eda_cfg.leakage_corr_threshold
        findings: list[str] = []
        corr_rows: list[dict[str, object]] = []

        target_numeric = target_series_numeric(df, data_cfg)
        numeric_cols = numeric_feature_names(df, data_cfg)
        if target_numeric is not None and numeric_cols:
            correlations = df[numeric_cols].astype(float).corrwith(target_numeric).dropna()
            for feature, value in correlations.items():
                if abs(float(value)) > threshold:
                    corr_rows.append(
                        {"feature": feature, "abs_corr_with_target": round(abs(float(value)), 4)}
                    )
                    findings.append(
                        f"LEAKAGE WARNING: numeric feature '{feature}' has "
                        f"|corr|={abs(float(value)):.4f} with target '{data_cfg.target}' "
                        f"(threshold {threshold:g}). Verify it is truly available at "
                        "prediction time, otherwise DROP IT."
                    )

        # A feature whose factorized codes match the target's is an exact copy of
        # the target up to relabeling (e.g. 0/1 vs "no"/"yes"), for any dtype.
        target_codes, _ = pd.factorize(df[data_cfg.target])
        duplicate_cols: list[str] = []
        for spec in data_cfg.feature_columns():
            if spec.name not in df.columns:
                continue
            feature_codes, _ = pd.factorize(df[spec.name])
            if (feature_codes == target_codes).all():
                duplicate_cols.append(spec.name)
                findings.append(
                    f"LEAKAGE WARNING: feature '{spec.name}' is IDENTICAL to the target "
                    f"'{data_cfg.target}' (up to relabeling). DROP IT before training."
                )

        if not findings:
            html = (
                "<p>No leakage indicators detected "
                f"(|corr| threshold {threshold:g}; no target duplicates).</p>"
            )
            return EdaSection(title, html)

        parts = [
            '<p class="alert">Potential target leakage detected — review these columns '
            "before training.</p>"
        ]
        if corr_rows:
            parts.append(df_to_html_table(pd.DataFrame(corr_rows)))
        if duplicate_cols:
            duplicates = ", ".join(f"'{name}'" for name in duplicate_cols)
            parts.append(f"<p>Duplicates of the target: {duplicates}.</p>")
        return EdaSection(title, "".join(parts), findings)
