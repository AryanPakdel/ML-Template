"""k-nearest neighbors for classification and regression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("knn")
class KnnModel(SklearnModel):
    """Instance-based k-NN; no training phase, sensitive to feature scaling."""

    name: ClassVar[str] = "knn"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def _build_estimator(self) -> KNeighborsClassifier | KNeighborsRegressor:
        """Return a fresh k-NN estimator matching the task (k-NN takes no seed)."""
        if self.task is TaskType.CLASSIFICATION:
            return KNeighborsClassifier(**self.params)
        return KNeighborsRegressor(**self.params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune neighborhood size, vote weighting, and Minkowski power."""
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
