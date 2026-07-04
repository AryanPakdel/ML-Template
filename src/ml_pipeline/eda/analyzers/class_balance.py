"""Class-balance analyzer: target label counts and shares (classification only)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from ml_pipeline.config.schema import DataConfig, EdaConfig
from ml_pipeline.core.types import TaskType
from ml_pipeline.eda.base import (
    EDA_REGISTRY,
    EdaAnalyzer,
    EdaSection,
    df_to_html_table,
    fig_to_html,
)

logger = logging.getLogger(__name__)

_MINORITY_PCT: float = 20.0
"""Minority-class share below which imbalance handling is suggested."""


@EDA_REGISTRY.register("class_balance")
class ClassBalanceAnalyzer(EdaAnalyzer):
    """Tabulate and plot target class frequencies; not applicable to regression."""

    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Build the class-balance section, or return ``None`` for regression."""
        if data_cfg.task is not TaskType.CLASSIFICATION:
            return None
        title = "Class balance"
        if data_cfg.target not in df.columns:
            logger.warning(
                "class_balance: target '%s' not in DataFrame; section skipped.",
                data_cfg.target,
            )
            return None

        counts = df[data_cfg.target].value_counts(dropna=False)
        total = int(counts.sum())
        rows: list[dict[str, object]] = []
        for label, count in counts.items():
            pct = 100.0 * int(count) / total if total else 0.0
            rows.append({"class": str(label), "count": int(count), "pct": f"{pct:.1f}%"})

        fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * len(counts)), 3.5))
        ax.bar([str(label) for label in counts.index], counts.to_numpy())
        ax.set_title(f"Class counts for '{data_cfg.target}'")
        ax.set_ylabel("count")
        fig.tight_layout()

        findings: list[str] = []
        if total and len(counts) > 1:
            minority_pct = 100.0 * int(counts.min()) / total
            if minority_pct < _MINORITY_PCT:
                minority_label = counts.idxmin()
                findings.append(
                    f"Minority class '{minority_label}' covers only {minority_pct:.1f}% of "
                    "rows — consider imbalance handling (model.imbalance: class_weight/"
                    "smote/random_over/random_under)."
                )

        html = df_to_html_table(pd.DataFrame(rows)) + fig_to_html(fig)
        return EdaSection(title, html, findings)
