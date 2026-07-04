"""The model contract every implementation satisfies — classical, deep, or ensemble.

Models receive **post-preprocessing** feature matrices (pandas DataFrames with
resolved feature names, or numpy arrays) and never see raw dataset columns, which
keeps them dataset-agnostic. The trainer, tuner, leaderboard, ensembles, SHAP
dispatch, and serving all program against this interface only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import pandas as pd

from ml_pipeline.core.registry import Registry
from ml_pipeline.core.types import ExplainerHint, TaskType

if TYPE_CHECKING:
    import optuna

ArrayLike = np.ndarray | pd.DataFrame

MODEL_REGISTRY: Registry[type[BaseModel]] = Registry("model")


class BaseModel(ABC):
    """Abstract model with a unified fit/predict/predict_proba interface.

    Class attributes:
        name: registry key, set by each implementation.
        supported_tasks: which :class:`TaskType` values the model handles.
        explainer_hint: which SHAP explainer family suits this model.

    Instances must be picklable with ``joblib`` so they can live inside the
    persisted :class:`~ml_pipeline.core.artifacts.PipelineBundle`. Deep-learning
    models override ``__getstate__``/``__setstate__`` to serialize weights.
    """

    name: ClassVar[str] = ""
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def __init__(self, params: dict[str, Any], task: TaskType, seed: int = 42) -> None:
        """Store run context; heavy objects are built lazily in :meth:`fit`.

        Args:
            params: implementation-specific hyperparameters (from config/Optuna).
            task: classification or regression.
            seed: random seed forwarded to the underlying estimator.

        Raises:
            ValueError: if the model does not support ``task``.
        """
        if task not in self.supported_tasks:
            raise ValueError(
                f"Model '{self.name}' does not support task '{task.value}' "
                f"(supported: {sorted(t.value for t in self.supported_tasks)})"
            )
        self.params: dict[str, Any] = dict(params)
        self.task = task
        self.seed = seed
        self.feature_names_: list[str] | None = None
        self.n_classes_: int | None = None

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Train on ``(X, y)``; ``(X_val, y_val)`` enables early stopping where supported.

        For classification, ``y`` is label-encoded to ``0..n_classes-1`` by the
        trainer before it reaches the model.
        """

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict encoded class indices (classification) or values (regression)."""

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Class probabilities of shape ``(n_samples, n_classes)``.

        Raises:
            NotImplementedError: for models without probability estimates
                (e.g. regressors); callers must handle this.
        """
        raise NotImplementedError(f"'{self.name}' does not implement predict_proba")

    def get_feature_importance(self) -> np.ndarray | None:
        """Per-feature importance aligned with ``feature_names_``, or ``None``."""
        return None

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Sample this model's default Optuna search space; ``{}`` = not tunable."""
        return {}

    def _remember_features(self, X: ArrayLike) -> None:
        """Capture feature names (when ``X`` is a DataFrame) for importances/SHAP."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task.value}, params={self.params})"
