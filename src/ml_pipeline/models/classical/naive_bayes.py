"""Gaussian naive Bayes for classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.naive_bayes import GaussianNB

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("naive_bayes")
class NaiveBayesModel(SklearnModel):
    """Gaussian NB; near-instant probabilistic baseline for classification."""

    name: ClassVar[str] = "naive_bayes"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset({TaskType.CLASSIFICATION})
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def _build_estimator(self) -> GaussianNB:
        """Return a fresh :class:`GaussianNB` configured from ``self.params``."""
        return GaussianNB(**self.params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune the variance-smoothing stabilizer on a log scale."""
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
        }
