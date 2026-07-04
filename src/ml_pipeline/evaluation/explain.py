"""Model interpretability: SHAP with automatic explainer dispatch, LIME fallback.

The explainer family is chosen from :attr:`BaseModel.explainer_hint`
(TREE/LINEAR/KERNEL/NONE); anything that fails degrades gracefully —
SHAP error -> LIME -> skip with a warning. Explainability never fails a run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_pipeline.config.schema import ExplainabilityConfig
from ml_pipeline.core.types import ExplainerHint, TaskType

logger = logging.getLogger(__name__)

_KERNEL_EVAL_CAP = 50  # KernelExplainer is O(background * eval * features)


def explain_model(
    model: Any,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    task: TaskType,
    output_dir: Path,
    cfg: ExplainabilityConfig,
    seed: int = 42,
) -> list[Path]:
    """Produce SHAP summary/dependence plots (or LIME HTML fallback) for a model.

    Args:
        model: fitted BaseModel.
        X_background: post-preprocessing training matrix (background/reference).
        X_eval: post-preprocessing evaluation matrix to explain.
        task: task type (drives the prediction function used).
        output_dir: where plot files are written.
        cfg: explainability settings (method, sample caps, top features).
        seed: sampling seed.

    Returns:
        Paths of the artifacts written (empty when skipped or failed).
    """
    if cfg.method == "none":
        return []
    hint = getattr(model, "explainer_hint", ExplainerHint.KERNEL)
    if cfg.method == "auto" and hint == ExplainerHint.NONE:
        logger.info("Model '%s' opts out of explainability; skipping", model.name)
        return []

    rng = np.random.default_rng(seed)
    X_bg = _sample(X_background, cfg.max_samples, rng)
    X_ev = _sample(X_eval, cfg.max_samples, rng)

    if cfg.method in ("auto", "shap"):
        try:
            return _shap_plots(model, hint, X_bg, X_ev, task, output_dir, cfg)
        except Exception as err:  # noqa: BLE001 - fall back, never fail the run
            logger.warning("SHAP failed (%s: %s); trying LIME", type(err).__name__, err)
    if cfg.method in ("auto", "lime"):
        try:
            return _lime_fallback(model, X_bg, X_ev, task, output_dir)
        except Exception as err:  # noqa: BLE001
            logger.warning("LIME failed too (%s: %s); skipping explainability", type(err).__name__, err)
    return []


def _sample(X: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random row subsample (without replacement) capped at ``n``."""
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[np.sort(idx)]


def _predict_fn(model: Any, task: TaskType):
    """Scalar prediction function for kernel/LIME explainers."""
    if task == TaskType.CLASSIFICATION:
        return lambda data: np.asarray(model.predict_proba(data))
    return lambda data: np.asarray(model.predict(data))


def _shap_plots(
    model: Any,
    hint: ExplainerHint,
    X_bg: pd.DataFrame,
    X_ev: pd.DataFrame,
    task: TaskType,
    output_dir: Path,
    cfg: ExplainabilityConfig,
) -> list[Path]:
    """Compute SHAP values with the hinted explainer and save summary/dependence plots."""
    import shap  # heavy import kept local

    estimator = getattr(model, "estimator", None)
    if hint == ExplainerHint.TREE and estimator is not None:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(X_ev)
    elif hint == ExplainerHint.LINEAR and estimator is not None:
        explainer = shap.LinearExplainer(estimator, X_bg)
        values = explainer.shap_values(X_ev)
    else:
        background = shap.sample(X_bg, min(50, len(X_bg)), random_state=0)
        X_ev = X_ev.iloc[: min(_KERNEL_EVAL_CAP, len(X_ev))]
        fn = _predict_fn(model, task)
        if task == TaskType.CLASSIFICATION:
            proba_fn = fn
            fn = lambda data: proba_fn(data)[:, -1]  # noqa: E731 - positive class
        explainer = shap.KernelExplainer(fn, background)
        values = explainer.shap_values(X_ev, silent=True)

    values = _to_matrix(values)
    paths: list[Path] = []

    summary_path = output_dir / "shap_summary.png"
    plt.figure()
    shap.summary_plot(values, X_ev, show=False, max_display=cfg.top_features)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    paths.append(summary_path)
    logger.info("Saved %s", summary_path)

    top = np.argsort(np.abs(values).mean(axis=0))[::-1][:3]
    for feature_idx in top:
        name = str(X_ev.columns[feature_idx])
        dep_path = output_dir / f"shap_dependence_{_safe(name)}.png"
        plt.figure()
        shap.dependence_plot(
            feature_idx, values, X_ev, interaction_index=None, show=False
        )
        plt.tight_layout()
        plt.savefig(dep_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        paths.append(dep_path)
    return paths


def _to_matrix(values: Any) -> np.ndarray:
    """Normalize SHAP output variants to a 2D (samples, features) matrix.

    Tree explainers return, depending on model/version: a 2D array, a list of
    per-class arrays, or a 3D (samples, features, classes) array. For
    classification we keep the positive/last class.
    """
    if isinstance(values, list):
        values = values[-1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _lime_fallback(
    model: Any,
    X_bg: pd.DataFrame,
    X_ev: pd.DataFrame,
    task: TaskType,
    output_dir: Path,
    n_rows: int = 3,
) -> list[Path]:
    """Explain a few evaluation rows with LIME, saving one HTML per row."""
    from lime.lime_tabular import LimeTabularExplainer

    mode = "classification" if task == TaskType.CLASSIFICATION else "regression"
    explainer = LimeTabularExplainer(
        X_bg.to_numpy(),
        feature_names=list(X_bg.columns),
        mode=mode,
        discretize_continuous=True,
    )
    fn = _predict_fn(model, task)
    paths: list[Path] = []
    for i in range(min(n_rows, len(X_ev))):
        explanation = explainer.explain_instance(X_ev.iloc[i].to_numpy(), fn)
        path = output_dir / f"lime_row_{i}.html"
        explanation.save_to_file(str(path))
        paths.append(path)
    logger.info("Saved %d LIME explanations to %s", len(paths), output_dir)
    return paths


def _safe(name: str) -> str:
    """Filesystem-safe feature name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
