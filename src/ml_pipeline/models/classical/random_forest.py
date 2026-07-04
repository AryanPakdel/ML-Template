"""Random forest for classification and regression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("random_forest")
class RandomForestModel(SklearnModel):
    """Bagged decision-tree ensemble; robust default for tabular data."""

    name: ClassVar[str] = "random_forest"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.TREE

    def _build_estimator(self) -> RandomForestClassifier | RandomForestRegressor:
        """Return a fresh forest estimator matching the task, seeded for determinism."""
        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        if self.task is TaskType.CLASSIFICATION:
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune ensemble size, tree complexity, and per-split feature sampling."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }
