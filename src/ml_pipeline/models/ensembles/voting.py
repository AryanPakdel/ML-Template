"""Voting ensemble over registered BaseModels (soft voting with hard fallback).

Implemented on our own contract rather than sklearn's ``VotingClassifier`` so
that any registered model — including the Lightning MLP — can participate.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Self

import numpy as np

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY, ArrayLike, BaseModel

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("voting")
class VotingEnsemble(BaseModel):
    """Average (optionally weighted) predictions of independently trained bases.

    params:
        base_models: list of registry keys (required).
        base_params: optional dict of per-base hyperparameters.
        weights: optional per-base weights (default: equal).
    """

    name: ClassVar[str] = "voting"
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Train every base model on the same (already transformed) data."""
        base_names: list[str] = list(self.params.get("base_models", []))
        if not base_names:
            raise ValueError("voting ensemble requires params.base_models (list of names)")
        base_params: dict[str, dict[str, Any]] = self.params.get("base_params", {})

        self._remember_features(X)
        self.models_: list[BaseModel] = []
        for i, base_name in enumerate(base_names):
            cls = MODEL_REGISTRY.get(base_name)
            if self.task not in cls.supported_tasks:
                raise ValueError(f"Base model '{base_name}' does not support {self.task.value}")
            model = cls(dict(base_params.get(base_name, {})), self.task, self.seed + i)
            logger.info("Voting: fitting base '%s'", base_name)
            model.fit(X, y, X_val, y_val, sample_weight=sample_weight)
            self.models_.append(model)

        weights = self.params.get("weights")
        self.weights_ = np.asarray(
            weights if weights is not None else [1.0] * len(self.models_), dtype=float
        )
        if len(self.weights_) != len(self.models_):
            raise ValueError("params.weights length must match params.base_models")
        self.weights_ = self.weights_ / self.weights_.sum()
        if self.task == TaskType.CLASSIFICATION:
            self.n_classes_ = int(np.max(y)) + 1
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Soft vote via averaged probabilities; hard majority vote as fallback."""
        if self.task == TaskType.REGRESSION:
            preds = np.stack([m.predict(X) for m in self.models_])
            return np.average(preds, axis=0, weights=self.weights_)
        try:
            return np.argmax(self.predict_proba(X), axis=1)
        except NotImplementedError:
            preds = np.stack([m.predict(X) for m in self.models_]).astype(int)
            n_classes = int(preds.max()) + 1
            votes = np.zeros((preds.shape[1], n_classes))
            for model_idx in range(preds.shape[0]):
                votes[np.arange(preds.shape[1]), preds[model_idx]] += self.weights_[model_idx]
            return np.argmax(votes, axis=1)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Weighted mean of base probabilities."""
        try:
            probas = [m.predict_proba(X) for m in self.models_]
        except NotImplementedError as err:
            raise NotImplementedError(
                f"voting: a base model lacks predict_proba ({err}); soft voting unavailable"
            ) from err
        return np.average(np.stack(probas), axis=0, weights=self.weights_)
