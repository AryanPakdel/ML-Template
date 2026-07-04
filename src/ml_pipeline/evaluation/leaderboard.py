"""Model comparison: train every configured model, rank them, explain the winner.

Ensembles (voting/stacking) are ordinary registered models, so they flow through
the exact same trainer path as everything else.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.core.artifacts import PipelineBundle, create_run_dir, new_run_id
from ml_pipeline.evaluation.metrics import METRIC_DIRECTIONS, resolve_primary_metric
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.tracking import mlflow_utils
from ml_pipeline.training.trainer import PipelineTrainer, TrainResult, load_validated_frame

logger = logging.getLogger(__name__)


def run_comparison(cfg: PipelineConfig) -> tuple[pd.DataFrame, Path]:
    """Train all ``compare.models`` (+ optional ensembles), producing a leaderboard.

    Returns:
        ``(leaderboard, output_dir)`` — the ranked DataFrame and the directory
        holding ``leaderboard.csv`` / ``leaderboard.md``.
    """
    metric, direction = resolve_primary_metric(cfg.data.task, cfg.tuning.metric)
    names = list(cfg.compare.models) or [cfg.model.name]

    jobs: list[tuple[str, dict[str, Any]]] = []
    for name in names:
        jobs.append((name, dict(cfg.compare.model_params.get(name, {}))))

    ensemble = cfg.compare.ensemble
    base_params = {n: dict(cfg.compare.model_params.get(n, {})) for n in ensemble.base_models}
    if ensemble.voting:
        jobs.append(("voting", {"base_models": ensemble.base_models, "base_params": base_params}))
    if ensemble.stacking:
        jobs.append(("stacking", {"base_models": ensemble.base_models, "base_params": base_params}))

    rows: list[dict[str, Any]] = []
    results: dict[str, TrainResult] = {}
    for name, params in jobs:
        model_cls = MODEL_REGISTRY.get(name)
        if cfg.data.task not in model_cls.supported_tasks:
            logger.warning("Skipping '%s': unsupported for task '%s'", name, cfg.data.task.value)
            continue
        logger.info("=== Leaderboard: training '%s' ===", name)
        result = PipelineTrainer(
            cfg, model_name=name, model_params=params, run_cv=True, explain=False
        ).run()
        results[name] = result

        row: dict[str, Any] = {"model": name, "run_id": result.run_id}
        cv = result.metrics.get("cv", {})
        row[f"cv_{metric}_mean"] = cv.get(f"cv_{metric}_mean")
        row[f"cv_{metric}_std"] = cv.get(f"cv_{metric}_std")
        for key, value in sorted(result.metrics.get("val", {}).items()):
            row[f"val_{key}"] = value
        row[f"test_{metric}"] = result.metrics.get("test", {}).get(metric)
        rows.append(row)

    if not rows:
        raise ValueError("No models were trained — check compare.models against the task.")

    leaderboard = pd.DataFrame(rows)
    ascending = METRIC_DIRECTIONS.get(metric, "maximize") == "minimize"
    leaderboard = leaderboard.sort_values(f"val_{metric}", ascending=ascending).reset_index(
        drop=True
    )

    out_dir = create_run_dir(cfg.run.artifacts_dir, new_run_id(cfg.run.experiment_name, "compare"))
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    (out_dir / "leaderboard.md").write_text(_markdown_table(leaderboard), encoding="utf-8")
    logger.info("Leaderboard (top: %s):\n%s", leaderboard.iloc[0]["model"], leaderboard.to_string())

    best_name = str(leaderboard.iloc[0]["model"])
    _explain_best(cfg, results[best_name], out_dir)

    tracking = mlflow_utils.init_tracking(cfg.run.mlflow, cfg.run.experiment_name)
    with mlflow_utils.start_run(out_dir.name, enabled=tracking, tags={"kind": "compare"}):
        mlflow_utils.log_artifacts_dir(out_dir, tracking)

    return leaderboard, out_dir


def _explain_best(cfg: PipelineConfig, best: TrainResult, out_dir: Path) -> None:
    """SHAP/LIME artifacts for the leaderboard winner (best-effort)."""
    if cfg.evaluation.explainability.method == "none":
        return
    try:
        from ml_pipeline.data.splitters import split_dataset
        from ml_pipeline.evaluation.explain import explain_model

        bundle = PipelineBundle.load(best.bundle_path)
        df = load_validated_frame(cfg)
        splits = split_dataset(df, cfg.data, cfg.run.seed)
        X_bg = bundle.transform(splits.train[bundle.metadata.feature_columns])
        X_ev = bundle.transform(splits.test[bundle.metadata.feature_columns])
        paths = explain_model(
            bundle.model,
            X_bg,
            X_ev,
            cfg.data.task,
            out_dir,
            cfg.evaluation.explainability,
            seed=cfg.run.seed,
        )
        if paths:
            logger.info("Explained leaderboard winner '%s' (%d artifacts)", best.model_name, len(paths))
    except Exception:  # noqa: BLE001 - never fail the comparison over explainability
        logger.warning("Winner explainability failed", exc_info=True)


def _markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Hand-rolled GitHub-flavored markdown table (avoids a tabulate dependency)."""

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return float_fmt.format(value)
        return "" if value is None else str(value)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body]) + "\n"
