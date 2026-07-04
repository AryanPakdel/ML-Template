"""Contracts, registry, and shared helpers for the automated EDA stage.

Analyzers are small classes registered in :data:`EDA_REGISTRY`; each turns a raw
DataFrame plus config into an :class:`EdaSection` (self-contained HTML fragment
plus plain-text findings), or ``None`` when it does not apply to the dataset or
task (e.g. class balance on regression). The report builder only consumes
sections, so adding an analyzer never touches orchestration code.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless backend; set before any pyplot rendering

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ml_pipeline.core.registry import Registry  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ml_pipeline.config.schema import DataConfig, EdaConfig

logger = logging.getLogger(__name__)

FIG_DPI: int = 110
"""Resolution used when rasterizing figures into the HTML report."""

NUMERIC_DTYPES: frozenset[str] = frozenset({"int", "float"})
"""``ColumnSpec.dtype`` values treated as numeric by the analyzers."""

CATEGORICAL_DTYPES: frozenset[str] = frozenset({"category", "string"})
"""``ColumnSpec.dtype`` values treated as categorical/string by the analyzers."""


@dataclass
class EdaSection:
    """One rendered report section produced by a single analyzer.

    Attributes:
        title: section heading shown in the report and table of contents.
        html: section body; may embed ``<img>`` tags with base64 PNG figures.
        findings: plain-text bullets surfaced in the markdown summary.
    """

    title: str
    html: str
    findings: list[str] = field(default_factory=list)


class EdaAnalyzer(ABC):
    """Contract for one EDA analyzer.

    Implementations are stateless: all inputs arrive through ``analyze`` so the
    same instance can be reused across datasets.
    """

    @abstractmethod
    def analyze(
        self, df: pd.DataFrame, data_cfg: DataConfig, eda_cfg: EdaConfig
    ) -> EdaSection | None:
        """Analyze ``df`` and return a report section.

        Args:
            df: raw (validated) dataset including the target column.
            data_cfg: dataset definition (columns, dtypes, target, task).
            eda_cfg: EDA thresholds and plot limits.

        Returns:
            An :class:`EdaSection`, or ``None`` when the analyzer is not
            applicable (the report simply omits the section).
        """


EDA_REGISTRY: Registry[type[EdaAnalyzer]] = Registry("eda_analyzer")


def fig_to_html(fig: Figure) -> str:
    """Render a matplotlib figure to an inline base64 PNG ``<img>`` tag.

    The figure is always closed afterwards to avoid leaking Agg canvases in
    long-running processes.

    Args:
        fig: the figure to rasterize.

    Returns:
        A self-contained ``<img>`` tag with a ``data:image/png;base64`` source.
    """
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=FIG_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="figure" />'


def subplot_grid(
    n_plots: int, ncols: int = 3, subplot_size: tuple[float, float] = (4.5, 3.2)
) -> tuple[Figure, list[Axes]]:
    """Create a grid figure with exactly ``n_plots`` visible axes.

    Args:
        n_plots: number of axes needed (must be >= 1).
        ncols: maximum number of columns in the grid.
        subplot_size: (width, height) in inches per subplot.

    Returns:
        The figure and a flat list of ``n_plots`` axes (surplus axes hidden).
    """
    ncols = max(1, min(ncols, n_plots))
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
        squeeze=False,
    )
    flat = list(axes.ravel())
    for ax in flat[n_plots:]:
        ax.set_visible(False)
    return fig, flat[:n_plots]


def numeric_feature_names(df: pd.DataFrame, data_cfg: DataConfig) -> list[str]:
    """Names of numeric feature columns (per config dtype) present in ``df``."""
    return [
        spec.name
        for spec in data_cfg.feature_columns()
        if spec.dtype in NUMERIC_DTYPES and spec.name in df.columns
    ]


def categorical_feature_names(df: pd.DataFrame, data_cfg: DataConfig) -> list[str]:
    """Names of categorical/string feature columns (per config dtype) in ``df``."""
    return [
        spec.name
        for spec in data_cfg.feature_columns()
        if spec.dtype in CATEGORICAL_DTYPES and spec.name in df.columns
    ]


def target_series_numeric(df: pd.DataFrame, data_cfg: DataConfig) -> pd.Series | None:
    """The target as a numeric series, factorizing categorical targets.

    Args:
        df: raw dataset.
        data_cfg: dataset definition holding the target name.

    Returns:
        A float series aligned with ``df`` (missing values stay ``NaN``), or
        ``None`` when the target column is absent from ``df``.
    """
    if data_cfg.target not in df.columns:
        logger.warning("Target column '%s' not found in DataFrame.", data_cfg.target)
        return None
    target = df[data_cfg.target]
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(float)
    codes, _uniques = pd.factorize(target)
    series = pd.Series(codes, index=target.index, name=target.name).astype(float)
    return series.mask(codes == -1)  # factorize encodes NaN as -1


def df_to_html_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a small DataFrame as an escaped, styled HTML table.

    Args:
        frame: tabular content to render.
        max_rows: optional row cap (a truncation note is appended when hit).

    Returns:
        An HTML ``<table class="eda-table">`` fragment.
    """
    note = ""
    if max_rows is not None and len(frame) > max_rows:
        note = f"<p><em>Showing first {max_rows} of {len(frame)} rows.</em></p>"
        frame = frame.head(max_rows)
    table = frame.to_html(
        index=False,
        border=0,
        classes="eda-table",
        na_rep="",
        float_format=lambda value: f"{value:,.4g}",
    )
    return f"{table}{note}"
