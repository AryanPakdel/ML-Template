"""Thin MLflow helpers: URI/experiment setup, safe logging, run context.

Only the fluent API (``mlflow.start_run``/``log_*``) is used — stable across
MLflow 2.x/3.x. All helpers no-op when tracking is disabled and never let a
tracking failure break a training run.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow

from ml_pipeline.config.schema import MlflowConfig

logger = logging.getLogger(__name__)

_MAX_PARAM_VALUE_LEN = 450
_MAX_PARAMS = 180


def init_tracking(cfg: MlflowConfig, experiment_name: str) -> bool:
    """Point MLflow at the configured store and experiment.

    Relative ``file:`` URIs are resolved against the current working directory
    so artifacts land inside the project regardless of caller location.

    Returns:
        True when tracking is active, False when disabled or setup failed.
    """
    if not cfg.enabled:
        return False
    try:
        uri = cfg.tracking_uri
        if uri.startswith("file:") and not uri.startswith("file:///"):
            uri = "file://" + str(Path(uri.removeprefix("file:")).resolve())
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        return True
    except Exception:  # noqa: BLE001 - tracking must never sink a run
        logger.warning("MLflow setup failed; continuing without tracking", exc_info=True)
        return False


@contextlib.contextmanager
def start_run(
    run_name: str, enabled: bool = True, nested: bool = False, tags: dict[str, Any] | None = None
) -> Iterator[Any]:
    """Context manager yielding the active MLflow run, or ``None`` when disabled."""
    if not enabled:
        yield None
        return
    try:
        with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
            yield run
    except Exception:  # noqa: BLE001
        logger.warning("MLflow run '%s' failed to start; continuing untracked", run_name)
        yield None


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys (``model.params.max_depth``)."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def log_config(config_dict: dict[str, Any], enabled: bool = True) -> None:
    """Log a (nested) config dict as flattened MLflow params, safely truncated."""
    if not enabled:
        return
    try:
        flat = flatten_dict(config_dict)
        items = list(flat.items())[:_MAX_PARAMS]
        mlflow.log_params({k: str(v)[:_MAX_PARAM_VALUE_LEN] for k, v in items})
    except Exception:  # noqa: BLE001
        logger.warning("MLflow param logging failed", exc_info=True)


def log_metrics(metrics: dict[str, float], enabled: bool = True, step: int | None = None) -> None:
    """Log numeric metrics (non-numeric values are skipped)."""
    if not enabled:
        return
    try:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric, step=step)
    except Exception:  # noqa: BLE001
        logger.warning("MLflow metric logging failed", exc_info=True)


def log_artifacts_dir(path: str | Path, enabled: bool = True) -> None:
    """Attach every file under ``path`` to the active run."""
    if not enabled:
        return
    try:
        mlflow.log_artifacts(str(path))
    except Exception:  # noqa: BLE001
        logger.warning("MLflow artifact logging failed", exc_info=True)


def set_tags(tags: dict[str, Any], enabled: bool = True) -> None:
    """Set tags on the active run."""
    if not enabled:
        return
    try:
        mlflow.set_tags(tags)
    except Exception:  # noqa: BLE001
        logger.warning("MLflow tag logging failed", exc_info=True)
