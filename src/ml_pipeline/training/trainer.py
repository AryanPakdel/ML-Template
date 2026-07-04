"""The pipeline orchestrator: one call runs load -> validate -> split -> CV ->
final fit -> evaluate -> explain -> persist bundle, with MLflow tracking.

Every component (loader, splitter, preprocessor pieces, model, sampler) is
resolved from its registry by config key — the trainer composes, it never
implements stage logic itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.config.loader import dump_config
from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.core.artifacts import (
    BundleMetadata,
    PipelineBundle,
    create_run_dir,
    new_run_id,
)
from ml_pipeline.core.types import ColumnRole, TaskType
from ml_pipeline.data.loaders import load_dataframe
from ml_pipeline.data.splitters import split_dataset
from ml_pipeline.data.validation import validate_dataframe
from ml_pipeline.evaluation import error_analysis, plots
from ml_pipeline.evaluation.metrics import compute_metrics
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.tracking import mlflow_utils
from ml_pipeline.training.cross_validation import (
    CVResult,
    cross_validate,
    fit_fold,
    predict_proba_or_none,
    prepare_features_and_target,
)
from ml_pipeline.utils.io import ensure_dir, write_json
from ml_pipeline.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def load_validated_frame(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the configured dataset, schema-validate it, and drop role=drop columns.

    Shared entry point for the trainer, tuner, leaderboard, EDA, and evaluate CLI.
    """
    df = load_dataframe(cfg.data.source)
    df = validate_dataframe(df, cfg.data)
    drop_cols = [c for c in cfg.data.columns_by_role(ColumnRole.DROP) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Dropped role=drop columns: %s", drop_cols)
    return df


@dataclass
class TrainResult:
    """Everything a caller (CLI, tuner, leaderboard) needs from a training run."""

    run_id: str
    run_dir: Path
    bundle_path: Path
    model_name: str
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    cv: CVResult | None = None
    mlflow_run_id: str | None = None


class PipelineTrainer:
    """Config-driven end-to-end training for a single model.

    Args:
        cfg: validated pipeline config.
        model_name: registry key overriding ``cfg.model.name`` (leaderboard/tuner).
        model_params: hyperparameters overriding ``cfg.model.params``.
        run_cv: whether to cross-validate on train+val before the final fit.
        explain: whether to produce SHAP/LIME artifacts for the final model.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        model_name: str | None = None,
        model_params: dict[str, Any] | None = None,
        run_cv: bool = True,
        explain: bool = True,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name or cfg.model.name
        self.model_params = dict(cfg.model.params if model_params is None else model_params)
        self.run_cv = run_cv
        self.explain = explain

    # ------------------------------------------------------------------ api
    def run(self) -> TrainResult:
        """Execute the full training pipeline and return the run summary."""
        cfg = self.cfg
        set_global_seed(cfg.run.seed)

        model_cls = MODEL_REGISTRY.get(self.model_name)
        if cfg.data.task not in model_cls.supported_tasks:
            raise ValueError(
                f"Model '{self.model_name}' does not support task '{cfg.data.task.value}'"
            )

        df = self._load_and_validate()
        splits = split_dataset(df, cfg.data, cfg.run.seed)

        run_id = new_run_id(cfg.run.experiment_name, self.model_name)
        run_dir = create_run_dir(cfg.run.artifacts_dir, run_id)
        logger.info("Run %s -> %s", run_id, run_dir)

        tracking = mlflow_utils.init_tracking(cfg.run.mlflow, cfg.run.experiment_name)
        with mlflow_utils.start_run(
            run_id, enabled=tracking, tags={"model": self.model_name, "run_id": run_id}
        ) as active:
            mlflow_run_id = active.info.run_id if active is not None else None
            mlflow_utils.log_config(cfg.model_dump(mode="json"), tracking)

            cv_result: CVResult | None = None
            if self.run_cv:
                trainval = pd.concat([splits.train, splits.val])
                cv_result = cross_validate(cfg, trainval, self.model_name, self.model_params)
                mlflow_utils.log_metrics(cv_result.summary(), tracking)

            # Consistent label encoding across splits: derive labels from all rows.
            _, _, class_labels = prepare_features_and_target(df, cfg)
            X_tr, y_tr, _ = prepare_features_and_target(splits.train, cfg, class_labels)
            X_va, y_va, _ = prepare_features_and_target(splits.val, cfg, class_labels)
            X_te, y_te, _ = prepare_features_and_target(splits.test, cfg, class_labels)

            preprocessor, feature_pipeline, model = fit_fold(
                cfg, X_tr, y_tr, self.model_name, self.model_params, X_va, y_va
            )

            def transform(X: pd.DataFrame) -> pd.DataFrame:
                out = preprocessor.transform(X)
                return out if feature_pipeline is None else feature_pipeline.transform(out)

            metrics: dict[str, dict[str, float]] = {}
            predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
            for split_name, X_s, y_s in (("val", X_va, y_va), ("test", X_te, y_te)):
                X_t = transform(X_s)
                y_pred = model.predict(X_t)
                y_proba = predict_proba_or_none(model, X_t, cfg.data.task)
                metrics[split_name] = compute_metrics(cfg.data.task, y_s, y_pred, y_proba)
                predictions[split_name] = (y_pred, y_proba)
                logger.info(
                    "%s metrics: %s",
                    split_name,
                    ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics[split_name].items())),
                )
            if cv_result is not None:
                metrics["cv"] = cv_result.summary()

            self._save_plots(run_dir, model, y_te, *predictions["test"], class_labels)
            self._save_error_analysis(run_dir, splits.test, y_te, *predictions["test"])
            if self.explain:
                self._run_explainability(run_dir, model, transform(X_tr), transform(X_te))

            bundle = self._build_bundle(
                run_id, preprocessor, feature_pipeline, model, class_labels, metrics
            )
            bundle_path = bundle.save(run_dir)
            dump_config(cfg, run_dir / "config_resolved.yaml")
            write_json(metrics, run_dir / "metrics.json")

            flat = {
                f"{split}_{name}": value
                for split, section in metrics.items()
                if split != "cv"
                for name, value in section.items()
            }
            mlflow_utils.log_metrics(flat, tracking)
            mlflow_utils.log_artifacts_dir(run_dir, tracking)

        logger.info("Training complete: %s", bundle_path)
        return TrainResult(
            run_id=run_id,
            run_dir=run_dir,
            bundle_path=bundle_path,
            model_name=self.model_name,
            metrics=metrics,
            cv=cv_result,
            mlflow_run_id=mlflow_run_id,
        )

    # -------------------------------------------------------------- internals
    def _load_and_validate(self) -> pd.DataFrame:
        """Load, schema-validate, and prune the raw dataset."""
        return load_validated_frame(self.cfg)

    def _save_plots(
        self,
        run_dir: Path,
        model: Any,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        class_labels: list[Any] | None,
    ) -> None:
        """Task-appropriate evaluation plots for the held-out test split."""
        plot_dir = ensure_dir(run_dir / "plots")
        if self.cfg.data.task == TaskType.CLASSIFICATION:
            plots.plot_confusion_matrix(
                y_true, y_pred, class_labels, plot_dir / "confusion_matrix.png"
            )
            if y_proba is not None:
                plots.plot_roc_curve(y_true, y_proba, plot_dir / "roc_curve.png")
                plots.plot_pr_curve(y_true, y_proba, plot_dir / "pr_curve.png")
        else:
            plots.plot_residuals(y_true, y_pred, plot_dir / "residuals.png")
            plots.plot_predicted_vs_actual(
                y_true, y_pred, plot_dir / "predicted_vs_actual.png"
            )
        importance = model.get_feature_importance()
        if importance is not None and model.feature_names_:
            plots.plot_feature_importance(
                model.feature_names_, importance, plot_dir / "feature_importance.png"
            )

    def _save_error_analysis(
        self,
        run_dir: Path,
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> None:
        """Persist the worst-predicted test rows for manual inspection."""
        worst = error_analysis.worst_rows(
            test_df,
            y_true,
            y_pred,
            self.cfg.data.task,
            y_proba=y_proba,
            top_n=self.cfg.evaluation.error_analysis_top_n,
            id_columns=self.cfg.data.columns_by_role(ColumnRole.ID),
        )
        error_analysis.save_error_analysis(worst, run_dir / "error_analysis.csv")

    def _run_explainability(
        self, run_dir: Path, model: Any, X_background: pd.DataFrame, X_eval: pd.DataFrame
    ) -> None:
        """SHAP (or LIME fallback) artifacts; failures are logged, never fatal."""
        if self.cfg.evaluation.explainability.method == "none":
            return
        try:
            from ml_pipeline.evaluation.explain import explain_model  # heavy import

            explain_model(
                model,
                X_background,
                X_eval,
                self.cfg.data.task,
                ensure_dir(run_dir / "plots"),
                self.cfg.evaluation.explainability,
                seed=self.cfg.run.seed,
            )
        except Exception:  # noqa: BLE001 - explainability is best-effort by design
            logger.warning("Explainability failed; continuing without it", exc_info=True)

    def _build_bundle(
        self,
        run_id: str,
        preprocessor: Any,
        feature_pipeline: Any,
        model: Any,
        class_labels: list[Any] | None,
        metrics: dict[str, dict[str, float]],
    ) -> PipelineBundle:
        """Assemble the single persisted training->inference artifact."""
        cfg = self.cfg
        feature_specs = cfg.data.feature_columns()
        metadata = BundleMetadata(
            run_id=run_id,
            created_at=pd.Timestamp.now().isoformat(timespec="seconds"),
            experiment_name=cfg.run.experiment_name,
            task=cfg.data.task.value,
            target=cfg.data.target,
            model_name=self.model_name,
            feature_columns=[c.name for c in feature_specs],
            raw_feature_schema=[c.model_dump(mode="json") for c in feature_specs],
            class_labels=class_labels,
            metrics={k: v for k, v in metrics.items() if k in ("val", "test")},
            config=cfg.model_dump(mode="json"),
            package_version=__import__("ml_pipeline").__version__,
        )
        return PipelineBundle(preprocessor, feature_pipeline, model, metadata)
