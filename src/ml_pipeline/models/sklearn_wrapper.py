"""Generic adapter turning any sklearn-style estimator into a :class:`BaseModel`.

Concrete classical models only implement :meth:`_build_estimator` (and optionally
a search space / fit-kwargs hook) — fit/predict plumbing lives here once.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Self

import numpy as np

from ml_pipeline.models.base import ArrayLike, BaseModel


class SklearnModel(BaseModel):
    """Base class for models wrapping a scikit-learn-compatible estimator."""

    def __init__(self, params: dict[str, Any], task, seed: int = 42) -> None:
        super().__init__(params, task, seed)
        self._estimator: Any = None

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return a fresh unfitted estimator configured from ``self.params``."""

    def _fit_kwargs(
        self, X_val: ArrayLike | None, y_val: np.ndarray | None
    ) -> dict[str, Any]:
        """Extra kwargs for ``estimator.fit`` (boosting models add ``eval_set`` here)."""
        return {}

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Build and fit the wrapped estimator; forwards ``sample_weight`` if supported."""
        self._remember_features(X)
        self._estimator = self._build_estimator()

        fit_kwargs = self._fit_kwargs(X_val, y_val)
        if sample_weight is not None:
            fit_params = inspect.signature(self._estimator.fit).parameters
            if "sample_weight" in fit_params:
                fit_kwargs["sample_weight"] = sample_weight

        self._estimator.fit(X, y, **fit_kwargs)
        if hasattr(self._estimator, "classes_"):
            self.n_classes_ = len(self._estimator.classes_)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Delegate to the fitted estimator."""
        self._check_fitted()
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Delegate to the fitted estimator; raises if it lacks predict_proba."""
        self._check_fitted()
        if not hasattr(self._estimator, "predict_proba"):
            raise NotImplementedError(
                f"'{self.name}' ({type(self._estimator).__name__}) has no predict_proba"
            )
        return np.asarray(self._estimator.predict_proba(X))

    def get_feature_importance(self) -> np.ndarray | None:
        """Tree importances or |coef|, whichever the estimator exposes."""
        if self._estimator is None:
            return None
        if hasattr(self._estimator, "feature_importances_"):
            return np.asarray(self._estimator.feature_importances_)
        if hasattr(self._estimator, "coef_"):
            coef = np.asarray(self._estimator.coef_)
            return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        return None

    @property
    def estimator(self) -> Any:
        """The wrapped fitted estimator (used by the SHAP dispatcher)."""
        return self._estimator

    def _check_fitted(self) -> None:
        if self._estimator is None:
            raise RuntimeError(f"Model '{self.name}' is not fitted yet; call fit() first.")
