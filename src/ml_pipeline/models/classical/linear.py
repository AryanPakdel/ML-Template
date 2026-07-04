"""Linear models: logistic regression (classification) and OLS (regression)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from sklearn.linear_model import LinearRegression, LogisticRegression

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel

if TYPE_CHECKING:
    import optuna


@MODEL_REGISTRY.register("logistic_regression")
class LogisticRegressionModel(SklearnModel):
    """L2-regularized logistic regression; a strong, explainable baseline."""

    name: ClassVar[str] = "logistic_regression"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset({TaskType.CLASSIFICATION})
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.LINEAR

    def _build_estimator(self) -> LogisticRegression:
        """Return a fresh :class:`LogisticRegression` configured from ``self.params``."""
        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        return LogisticRegression(**params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """Tune the inverse regularization strength on a log scale."""
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}


@MODEL_REGISTRY.register("linear_regression")
class LinearRegressionModel(SklearnModel):
    """Ordinary least squares regression; the simplest regression baseline."""

    name: ClassVar[str] = "linear_regression"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset({TaskType.REGRESSION})
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.LINEAR

    def _build_estimator(self) -> LinearRegression:
        """Return a fresh :class:`LinearRegression` configured from ``self.params``."""
        return LinearRegression(**self.params)

    @classmethod
    def get_default_search_space(
        cls, trial: optuna.Trial, task: TaskType
    ) -> dict[str, Any]:
        """OLS has no hyperparameters worth tuning; empty space = not tunable."""
        return {}
