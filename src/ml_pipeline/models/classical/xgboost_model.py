"""XGBoost gradient boosting for classification and regression.

``xgboost`` is imported lazily inside :meth:`_build_estimator` so the registry
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


@MODEL_REGISTRY.register("xgboost")
class XgboostModel(SklearnModel):
    """Gradient-boosted trees via XGBoost; strong tabular performer."""

    name: ClassVar[str] = "xgboost"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.TREE

    def _build_estimator(self) -> Any:
        """Return a fresh ``XGBClassifier``/``XGBRegressor`` from ``self.params``."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Model 'xgboost' requires the 'xgboost' package: pip install xgboost"
            ) from exc

        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        params.setdefault("verbosity", 0)
        params.setdefault("n_jobs", -1)
        if self.task is TaskType.CLASSIFICATION:
            return XGBClassifier(**params)
        return XGBRegressor(**params)

    def _fit_kwargs(
        self, X_val: ArrayLike | None, y_val: np.ndarray | None
    ) -> dict[str, Any]:
        """Pass the validation split as ``eval_set`` so early stopping can engage."""
        if X_val is None:
            return {}
        return {"eval_set": [(X_val, y_val)], "verbose": False}

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune boosting length, tree shape, sampling, and L2 regularization."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
