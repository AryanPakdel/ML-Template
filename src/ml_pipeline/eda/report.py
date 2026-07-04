"""Assemble the automated EDA report (self-contained HTML + markdown summary).

``run_eda`` resolves the analyzer keys listed in ``cfg.eda.analyzers`` against
:data:`~ml_pipeline.eda.base.EDA_REGISTRY`, collects the applicable sections, and
renders two files into the output directory: ``report.html`` (inline CSS, base64
figures, table of contents) and ``report.md`` (findings-only summary).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pandas as pd

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.eda.base import EDA_REGISTRY, EdaSection
from ml_pipeline.utils.io import ensure_dir

logger = logging.getLogger(__name__)

HTML_REPORT_NAME = "report.html"
MD_REPORT_NAME = "report.md"

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1100px; padding: 2rem 1.5rem 4rem;
       color: #1f2430; background: #ffffff; line-height: 1.55; }
h1 { border-bottom: 3px solid #4c72b0; padding-bottom: 0.4rem; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #d8dce4; padding-bottom: 0.3rem; }
table.meta, table.eda-table { border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }
table.meta th, table.meta td,
table.eda-table th, table.eda-table td { border: 1px solid #d8dce4;
       padding: 0.35rem 0.7rem; text-align: left; }
table.meta th, table.eda-table thead th { background: #f0f3f8; }
table.eda-table tbody tr:nth-child(even) td { background: #fafbfd; }
img { max-width: 100%; height: auto; }
nav.toc { background: #f7f8fa; border: 1px solid #e0e3ea; border-radius: 8px;
       padding: 0.75rem 1.5rem; }
section { margin-bottom: 2.5rem; }
.findings { background: #fff7e6; border-left: 4px solid #e8a13c;
       padding: 0.75rem 1rem; margin: 1rem 0; }
.findings ul { margin: 0.4rem 0 0; padding-left: 1.4rem; }
.alert { color: #b00020; font-weight: 700; }
code { background: #f0f3f8; padding: 0.1rem 0.3rem; border-radius: 4px; }
"""


def _dataset_meta(df: pd.DataFrame, cfg: PipelineConfig, generated: str) -> list[tuple[str, str]]:
    """Header key/value pairs shared by the HTML and markdown reports."""
    return [
        ("Dataset", cfg.data.source.path),
        ("Task", cfg.data.task.value),
        ("Target", cfg.data.target),
        ("Shape", f"{df.shape[0]:,} rows x {df.shape[1]} columns"),
        ("Generated", generated),
    ]


def _render_html(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    sections: list[tuple[str, EdaSection]],
    generated: str,
) -> str:
    """Render the full self-contained HTML document."""
    meta_rows = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"
        for key, value in _dataset_meta(df, cfg, generated)
    )
    toc_items = "".join(
        f'<li><a href="#eda-{escape(key)}">{escape(section.title)}</a></li>'
        for key, section in sections
    )
    body_parts: list[str] = []
    for key, section in sections:
        findings_html = ""
        if section.findings:
            items = "".join(f"<li>{escape(finding)}</li>" for finding in section.findings)
            findings_html = (
                f'<div class="findings"><strong>Findings</strong><ul>{items}</ul></div>'
            )
        body_parts.append(
            f'<section id="eda-{escape(key)}">'
            f"<h2>{escape(section.title)}</h2>{findings_html}{section.html}</section>"
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"<title>EDA report — {escape(cfg.data.target)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        "<h1>EDA report</h1>\n"
        f'<table class="meta">{meta_rows}</table>\n'
        f'<nav class="toc"><strong>Contents</strong><ol>{toc_items}</ol></nav>\n'
        + "\n".join(body_parts)
        + "\n</body>\n</html>\n"
    )


def _render_markdown(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    sections: list[tuple[str, EdaSection]],
    generated: str,
) -> str:
    """Render the markdown summary: header plus per-section findings bullets."""
    lines: list[str] = ["# EDA report", ""]
    lines.extend(
        f"- **{key}:** {value}" for key, value in _dataset_meta(df, cfg, generated)
    )
    lines.append("")
    for _key, section in sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.findings:
            lines.extend(f"- {finding}" for finding in section.findings)
        else:
            lines.append("- No notable findings.")
        lines.append("")
    return "\n".join(lines)


def run_eda(df: pd.DataFrame, cfg: PipelineConfig, output_dir: Path) -> Path:
    """Run the configured analyzers and write ``report.html`` + ``report.md``.

    Args:
        df: raw (validated) dataset including the target column.
        cfg: full pipeline config; ``cfg.eda.analyzers`` selects the analyzers.
        output_dir: directory for the reports (created when missing).

    Returns:
        Path to the written ``report.html``.

    Raises:
        KeyError: when ``cfg.eda.analyzers`` names an unregistered analyzer.
    """
    sections: list[tuple[str, EdaSection]] = []
    for key in cfg.eda.analyzers:
        analyzer_cls = EDA_REGISTRY.get(key)
        section = analyzer_cls().analyze(df, cfg.data, cfg.eda)
        if section is None:
            logger.info("EDA analyzer '%s' not applicable; section skipped.", key)
            continue
        sections.append((key, section))
        logger.debug(
            "EDA analyzer '%s' produced section '%s' (%d findings).",
            key,
            section.title,
            len(section.findings),
        )

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_dir = ensure_dir(Path(output_dir))
    html_path = output_dir / HTML_REPORT_NAME
    md_path = output_dir / MD_REPORT_NAME
    html_path.write_text(_render_html(df, cfg, sections, generated), encoding="utf-8")
    md_path.write_text(_render_markdown(df, cfg, sections, generated), encoding="utf-8")
    logger.info("EDA report written: %s (html), %s (markdown)", html_path, md_path)
    return html_path
