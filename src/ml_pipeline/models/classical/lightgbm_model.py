"""LightGBM gradient boosting for classification and regression.

``lightgbm`` is imported lazily inside :meth:`_build_estimator` so the registry
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


@MODEL_REGISTRY.register("lightgbm")
class LightgbmModel(SklearnModel):
    """Gradient-boosted trees via LightGBM; fast training on wide tabular data."""

    name: ClassVar[str] = "lightgbm"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.TREE

    def _build_estimator(self) -> Any:
        """Return a fresh ``LGBMClassifier``/``LGBMRegressor`` from ``self.params``."""
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Model 'lightgbm' requires the 'lightgbm' package: pip install lightgbm"
            ) from exc

        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        params.setdefault("verbose", -1)
        if self.task is TaskType.CLASSIFICATION:
            return LGBMClassifier(**params)
        return LGBMRegressor(**params)

    def _fit_kwargs(
        self, X_val: ArrayLike | None, y_val: np.ndarray | None
    ) -> dict[str, Any]:
        """Pass the validation split as ``eval_set`` so early stopping can engage."""
        if X_val is None:
            return {}
        return {"eval_set": [(X_val, y_val)]}

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune boosting length, leaf-wise tree complexity, and sampling."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
