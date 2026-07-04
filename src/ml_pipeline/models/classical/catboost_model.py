"""CatBoost gradient boosting for classification and regression.

``catboost`` is imported lazily inside :meth:`_build_estimator` so the registry
stays importable in environments where the optional dependency is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY, ArrayLike
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("catboost")
class CatboostModel(SklearnModel):
    """Gradient-boosted trees via CatBoost; ordered boosting, strong defaults."""

    name: ClassVar[str] = "catboost"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.TREE

    def _build_estimator(self) -> Any:
        """Return a fresh ``CatBoostClassifier``/``CatBoostRegressor`` from ``self.params``."""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Model 'catboost' requires the 'catboost' package: pip install catboost"
            ) from exc

        params = dict(self.params)
        params.setdefault("random_seed", self.seed)
        params.setdefault("verbose", 0)
        params.setdefault("allow_writing_files", False)
        if self.task is TaskType.CLASSIFICATION:
            return CatBoostClassifier(**params)
        return CatBoostRegressor(**params)

    def _fit_kwargs(
        self, X_val: ArrayLike | None, y_val: np.ndarray | None
    ) -> dict[str, Any]:
        """Pass the validation split as ``eval_set`` so early stopping can engage."""
        if X_val is None:
            return {}
        return {"eval_set": (X_val, y_val)}

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune boosting length, learning rate, depth, and L2 leaf regularization."""
        return {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
