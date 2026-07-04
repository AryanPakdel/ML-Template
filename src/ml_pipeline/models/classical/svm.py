"""Support vector machines for classification (SVC) and regression (SVR)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.svm import SVC, SVR

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("svm")
class SvmModel(SklearnModel):
    """Kernel SVM; ``probability=True`` on SVC so predict_proba works downstream."""

    name: ClassVar[str] = "svm"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def _build_estimator(self) -> SVC | SVR:
        """Return a fresh :class:`SVC` or :class:`SVR` configured from ``self.params``."""
        params = dict(self.params)
        if self.task is TaskType.CLASSIFICATION:
            params.setdefault("probability", True)
            params.setdefault("random_state", self.seed)
            return SVC(**params)
        # SVR is deterministic and has no probability/random_state parameters.
        params.pop("probability", None)
        params.pop("random_state", None)
        return SVR(**params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune regularization, kernel width heuristic, and kernel family."""
        return {
            "C": trial.suggest_float("C", 1e-2, 100, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        }
