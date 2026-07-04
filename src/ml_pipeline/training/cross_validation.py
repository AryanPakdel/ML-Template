"""The pipeline's own cross-validation fold loop.

Per fold, the preprocessor and feature pipeline are re-fit on the training fold
only, then the fold model is trained on the transformed (optionally resampled)
data. This single loop guarantees zero leakage for target encoding, SMOTE,
importance-based selection — and works identically for classical and DL models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from ml_pipeline.config.schema import CvConfig, PipelineConfig
from ml_pipeline.core.registry import Registry
from ml_pipeline.core.types import TaskType
from ml_pipeline.evaluation.metrics import compute_metrics
from ml_pipeline.features.builder import build_feature_pipeline
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.preprocessing.builder import build_preprocessor
from ml_pipeline.training.imbalance import apply_sampler, balanced_sample_weight

logger = logging.getLogger(__name__)

# Factories: (cv_cfg, seed) -> sklearn CV splitter.
CV_SPLITTER_REGISTRY: Registry[Callable[[CvConfig, int], Any]] = Registry("cv_splitter")


@CV_SPLITTER_REGISTRY.register("kfold")
def _kfold(cfg: CvConfig, seed: int) -> KFold:
    return KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=seed if cfg.shuffle else None,
    )


@CV_SPLITTER_REGISTRY.register("stratified")
def _stratified(cfg: CvConfig, seed: int) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=seed if cfg.shuffle else None,
    )


@CV_SPLITTER_REGISTRY.register("timeseries")
def _timeseries(cfg: CvConfig, seed: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=cfg.n_splits)


@dataclass
class CVResult:
    """Per-fold and aggregated cross-validation metrics."""

    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def summary(self, prefix: str = "cv") -> dict[str, float]:
        """Flat ``{prefix}_{metric}_mean/std`` dict for logging."""
        out: dict[str, float] = {}
        for key, value in self.mean.items():
            out[f"{prefix}_{key}_mean"] = value
            out[f"{prefix}_{key}_std"] = self.std.get(key, 0.0)
        return out


def prepare_features_and_target(
    df: pd.DataFrame, cfg: PipelineConfig, class_labels: list[Any] | None = None
) -> tuple[pd.DataFrame, np.ndarray, list[Any] | None]:
    """Split a raw (validated) frame into model inputs and an encoded target.

    Args:
        df: validated rows including the target column.
        cfg: full pipeline config.
        class_labels: fixed label ordering to encode against (pass the labels
            derived from the full dataset so every split shares one mapping);
            derived from ``df`` when omitted.

    Returns:
        ``(X, y, class_labels)`` where ``X`` holds only role=feature columns,
        ``y`` is label-encoded to ``0..n-1`` for classification (``class_labels``
        maps encoded index back to the original label) and left numeric for
        regression (``class_labels`` is ``None``).
    """
    data_cfg = cfg.data
    feature_names = [c.name for c in data_cfg.feature_columns()]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing declared feature columns: {missing}")

    X = df[feature_names]
    y_raw = df[data_cfg.target]
    if data_cfg.task == TaskType.CLASSIFICATION:
        if class_labels is None:
            labels = np.unique(y_raw.to_numpy())
        else:
            labels = np.asarray(class_labels)
        y = np.searchsorted(labels, y_raw.to_numpy())
        if not np.array_equal(labels[y], y_raw.to_numpy()):
            unseen = sorted(set(y_raw) - set(labels.tolist()))
            raise ValueError(f"Target contains labels outside the known set: {unseen}")
        return X, y.astype(np.int64), labels.tolist()
    return X, y_raw.to_numpy(dtype=np.float64), None


def filter_training_rows(
    X: pd.DataFrame, y: np.ndarray, cfg: PipelineConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop numeric-outlier rows from *training* data when configured (method=remove).

    Bounds come from the same data being filtered (train fold / train split), so
    validation and test rows are never touched.
    """
    outlier_cfg = cfg.preprocessing.numeric.outliers
    if outlier_cfg.method != "remove":
        return X, y
    numeric_cols = [
        c.name
        for c in cfg.data.feature_columns()
        if c.dtype in ("int", "float") and c.name in X.columns
    ]
    if not numeric_cols:
        return X, y
    mask = pd.Series(True, index=X.index)
    for col in numeric_cols:
        series = X[col]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - outlier_cfg.iqr_factor * iqr, q3 + outlier_cfg.iqr_factor * iqr
        mask &= series.isna() | series.between(low, high)
    dropped = int((~mask).sum())
    if dropped:
        logger.info("Outlier removal: dropped %d/%d training rows", dropped, len(X))
    return X.loc[mask], y[mask.to_numpy()]


def fit_fold(
    cfg: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    model_params: dict[str, Any],
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[Any, Any, Any]:
    """Fit preprocessor + feature pipeline + model on one training set.

    Shared by the CV loop and the final fit so the two paths can never drift.

    Returns:
        ``(preprocessor, feature_pipeline, model)``, all fitted.
    """
    X_train, y_train = filter_training_rows(X_train, y_train, cfg)

    preprocessor = build_preprocessor(cfg.preprocessing, cfg.data, train_df=X_train)
    X_train_t = preprocessor.fit_transform(X_train, y_train)

    feature_pipeline = build_feature_pipeline(cfg.features, cfg.data.task, cfg.run.seed)
    if feature_pipeline is not None:
        X_train_t = feature_pipeline.fit_transform(X_train_t, y_train)

    X_val_t = None
    if X_val is not None and len(X_val):
        X_val_t = preprocessor.transform(X_val)
        if feature_pipeline is not None:
            X_val_t = feature_pipeline.transform(X_val_t)

    imbalance = cfg.model.imbalance
    sample_weight = balanced_sample_weight(imbalance.strategy, y_train)
    X_train_t, y_train = apply_sampler(
        imbalance.strategy, imbalance.options, cfg.run.seed, X_train_t, y_train
    )
    if sample_weight is not None and len(sample_weight) != len(y_train):
        sample_weight = balanced_sample_weight(imbalance.strategy, y_train)

    model_cls = MODEL_REGISTRY.get(model_name)
    model = model_cls(model_params, cfg.data.task, cfg.run.seed)
    model.fit(X_train_t, y_train, X_val_t, y_val, sample_weight=sample_weight)
    return preprocessor, feature_pipeline, model


def cross_validate(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> CVResult:
    """Run the configured CV strategy over ``df`` (train+val portion of the data).

    Args:
        cfg: full pipeline config.
        df: raw validated rows to cross-validate on.
        model_name/model_params: override ``cfg.model`` (used by tuner/leaderboard).
    """
    model_name = model_name or cfg.model.name
    model_params = dict(cfg.model.params if model_params is None else model_params)

    X, y, _ = prepare_features_and_target(df, cfg)
    splitter = CV_SPLITTER_REGISTRY.get(cfg.training.cv.strategy)(cfg.training.cv, cfg.run.seed)
    split_target = y if cfg.training.cv.strategy == "stratified" else None

    result = CVResult()
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, split_target)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        preprocessor, feature_pipeline, model = fit_fold(
            cfg, X_tr, y_tr, model_name, model_params, X_va, y_va
        )

        X_va_t = preprocessor.transform(X_va)
        if feature_pipeline is not None:
            X_va_t = feature_pipeline.transform(X_va_t)
        y_pred = model.predict(X_va_t)
        y_proba = predict_proba_or_none(model, X_va_t, cfg.data.task)

        fold = compute_metrics(cfg.data.task, y_va, y_pred, y_proba)
        result.fold_metrics.append(fold)
        logger.info("Fold %d/%d: %s", fold_idx + 1, cfg.training.cv.n_splits, _fmt(fold))

    keys = sorted({k for fold in result.fold_metrics for k in fold})
    for key in keys:
        values = [fold[key] for fold in result.fold_metrics if key in fold]
        result.mean[key] = float(np.mean(values))
        result.std[key] = float(np.std(values))
    logger.info("CV mean: %s", _fmt(result.mean))
    return result


def predict_proba_or_none(
    model: Any, X: pd.DataFrame | np.ndarray, task: TaskType
) -> np.ndarray | None:
    """Class probabilities when the model provides them, else ``None``."""
    if task != TaskType.CLASSIFICATION:
        return None
    try:
        return model.predict_proba(X)
    except NotImplementedError:
        return None


def _fmt(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
