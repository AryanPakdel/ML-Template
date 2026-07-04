"""Optuna hyperparameter search wrapping classical and DL models uniformly.

The objective merges each model's own default search space
(:meth:`BaseModel.get_default_search_space`) with config-declared overrides,
scores candidates with the shared leakage-safe CV loop, and logs every trial as
a nested MLflow run. The best parameters are then re-fit through the normal
:class:`PipelineTrainer` so tuning output is a standard run artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from ml_pipeline.config.schema import PipelineConfig, SearchSpaceParam
from ml_pipeline.data.splitters import split_dataset
from ml_pipeline.evaluation.metrics import resolve_primary_metric
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.tracking import mlflow_utils
from ml_pipeline.training.cross_validation import cross_validate
from ml_pipeline.training.trainer import PipelineTrainer, TrainResult, load_validated_frame
from ml_pipeline.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    """Outcome of a tuning study, including the re-fit best run."""

    model_name: str
    metric: str
    direction: str
    best_value: float
    best_params: dict[str, Any]
    n_trials: int
    best_run: TrainResult


def suggest_from_spec(trial: optuna.Trial, name: str, spec: SearchSpaceParam) -> Any:
    """Sample one config-declared search-space dimension."""
    if spec.type == "categorical":
        return trial.suggest_categorical(name, spec.choices)
    if spec.type == "int":
        return trial.suggest_int(
            name,
            int(spec.low),
            int(spec.high),
            log=spec.log,
            step=1 if spec.log or spec.step is None else int(spec.step),
        )
    return trial.suggest_float(
        name,
        spec.low,
        spec.high,
        log=spec.log,
        step=None if spec.log else spec.step,
    )


class OptunaTuner:
    """Config-driven Optuna study for the configured model."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def tune(self) -> TuneResult:
        """Run the study and re-fit the best parameters into a normal run."""
        cfg = self.cfg
        set_global_seed(cfg.run.seed)

        model_name = cfg.model.name
        model_cls = MODEL_REGISTRY.get(model_name)
        metric, direction = resolve_primary_metric(cfg.data.task, cfg.tuning.metric)
        if cfg.tuning.direction != "auto":
            direction = cfg.tuning.direction

        df = load_validated_frame(cfg)
        splits = split_dataset(df, cfg.data, cfg.run.seed)
        trainval = pd.concat([splits.train, splits.val])

        tracking = mlflow_utils.init_tracking(cfg.run.mlflow, cfg.run.experiment_name)

        def objective(trial: optuna.Trial) -> float:
            params = dict(model_cls.get_default_search_space(trial, cfg.data.task))
            for pname, spec in cfg.tuning.search_space.items():
                params[pname] = suggest_from_spec(trial, pname, spec)
            merged = {**cfg.model.params, **params}
            trial.set_user_attr(
                "model_params", {k: v for k, v in merged.items() if k != "callbacks"}
            )
            if model_name == "mlp":
                from ml_pipeline.tuning.callbacks import OptunaPruningCallback

                merged["callbacks"] = [OptunaPruningCallback(trial)]

            with mlflow_utils.start_run(
                f"trial_{trial.number}", enabled=tracking, nested=True
            ):
                mlflow_utils.log_config(trial.user_attrs["model_params"], tracking)
                cv = cross_validate(cfg, trainval, model_name, merged)
                mlflow_utils.log_metrics(cv.summary(), tracking)

            if metric not in cv.mean:
                raise ValueError(
                    f"Tuning metric '{metric}' was not produced by CV "
                    f"(got: {sorted(cv.mean)}). Pick one of those via tuning.metric."
                )
            return cv.mean[metric]

        pruner = (
            optuna.pruners.MedianPruner(n_warmup_steps=3)
            if cfg.tuning.pruner == "median"
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            study_name=f"{cfg.run.experiment_name}_{model_name}",
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=cfg.run.seed),
            pruner=pruner,
        )

        logger.info(
            "Tuning %s: %d trials, metric=%s (%s)",
            model_name,
            cfg.tuning.n_trials,
            metric,
            direction,
        )
        with mlflow_utils.start_run(f"tune_{model_name}", enabled=tracking):
            study.optimize(
                objective,
                n_trials=cfg.tuning.n_trials,
                timeout=cfg.tuning.timeout_s,
                show_progress_bar=False,
            )
            mlflow_utils.log_metrics({f"best_{metric}": study.best_value}, tracking)

        best_params = dict(study.best_trial.user_attrs["model_params"])
        logger.info("Best %s=%.5f with params %s", metric, study.best_value, best_params)

        best_run = PipelineTrainer(
            cfg, model_name=model_name, model_params=best_params, run_cv=False
        ).run()

        return TuneResult(
            model_name=model_name,
            metric=metric,
            direction=direction,
            best_value=float(study.best_value),
            best_params=best_params,
            n_trials=len(study.trials),
            best_run=best_run,
        )
