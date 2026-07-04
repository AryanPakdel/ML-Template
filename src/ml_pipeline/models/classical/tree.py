"""Single decision tree for classification and regression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("decision_tree")
class DecisionTreeModel(SklearnModel):
    """CART decision tree; fast, interpretable, and a useful overfitting probe."""

    name: ClassVar[str] = "decision_tree"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.TREE

    def _build_estimator(self) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """Return a fresh tree estimator matching the task, seeded for determinism."""
        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        if self.task is TaskType.CLASSIFICATION:
            return DecisionTreeClassifier(**params)
        return DecisionTreeRegressor(**params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune tree depth and leaf size, the main complexity controls."""
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        }
