"""Stacking ensemble: out-of-fold base predictions feed a linear meta-learner.

Base learners are any registered BaseModels; OOF construction happens inside
``fit`` on the already-transformed matrix, so the ensemble remains a drop-in
model for the trainer/leaderboard/serving.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Self

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY, ArrayLike, BaseModel

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("stacking")
class StackingEnsemble(BaseModel):
    """Two-level stack: OOF base predictions -> logistic/ridge meta-learner.

    params:
        base_models: list of registry keys (required).
        base_params: optional per-base hyperparameters.
        n_folds: OOF fold count (default 5).
    """

    name: ClassVar[str] = "stacking"
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Build OOF meta-features, fit the meta-learner, refit bases on all data."""
        base_names: list[str] = list(self.params.get("base_models", []))
        if not base_names:
            raise ValueError("stacking ensemble requires params.base_models (list of names)")
        base_params: dict[str, dict[str, Any]] = self.params.get("base_params", {})
        n_folds = int(self.params.get("n_folds", 5))

        self._remember_features(X)
        X_arr = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if self.task == TaskType.CLASSIFICATION:
            self.n_classes_ = int(np.max(y)) + 1
            folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        else:
            folds = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        def make_model(base_name: str, offset: int) -> BaseModel:
            cls = MODEL_REGISTRY.get(base_name)
            if self.task not in cls.supported_tasks:
                raise ValueError(f"Base model '{base_name}' does not support {self.task.value}")
            return cls(dict(base_params.get(base_name, {})), self.task, self.seed + offset)

        oof_blocks: list[np.ndarray] = []
        for i, base_name in enumerate(base_names):
            width = self.n_classes_ if self.task == TaskType.CLASSIFICATION else 1
            block = np.zeros((len(y), width))
            proba_ok = True
            logger.info("Stacking: OOF predictions for base '%s'", base_name)
            for train_idx, val_idx in folds.split(X_arr, y):
                model = make_model(base_name, i)
                model.fit(X_arr[train_idx], y[train_idx])
                block[val_idx] = self._base_outputs(model, X_arr[val_idx], width)
                proba_ok = proba_ok and getattr(model, "_last_proba_ok", True)
            oof_blocks.append(block)

        meta_X = np.hstack(oof_blocks)
        if self.task == TaskType.CLASSIFICATION:
            self.meta_ = LogisticRegression(max_iter=2000, random_state=self.seed)
        else:
            self.meta_ = RidgeCV()
        self.meta_.fit(meta_X, y)

        self.models_ = []
        for i, base_name in enumerate(base_names):
            model = make_model(base_name, i)
            model.fit(X_arr, y, sample_weight=sample_weight)
            self.models_.append(model)
        return self

    def _base_outputs(self, model: BaseModel, X: np.ndarray, width: int) -> np.ndarray:
        """Per-base meta-features: class probabilities or (fallback) predictions."""
        if self.task == TaskType.CLASSIFICATION:
            try:
                proba = model.predict_proba(X)
                if proba.shape[1] == width:
                    return proba
            except NotImplementedError:
                model._last_proba_ok = False  # noqa: SLF001 - internal bookkeeping
            preds = model.predict(X).astype(int)
            onehot = np.zeros((len(preds), width))
            onehot[np.arange(len(preds)), preds] = 1.0
            return onehot
        return model.predict(X).reshape(-1, 1)

    def _meta_features(self, X: ArrayLike) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        width = self.n_classes_ if self.task == TaskType.CLASSIFICATION else 1
        return np.hstack([self._base_outputs(m, X_arr, width) for m in self.models_])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Meta-learner prediction over base outputs."""
        return np.asarray(self.meta_.predict(self._meta_features(X)))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Meta-learner probabilities (classification only)."""
        if self.task != TaskType.CLASSIFICATION:
            raise NotImplementedError("predict_proba is classification-only")
        return np.asarray(self.meta_.predict_proba(self._meta_features(X)))
