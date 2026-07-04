"""Sklearn-compatible feature engineering/selection transformers.

Every transformer here operates on the **post-preprocessing** matrix — an
all-numeric pandas DataFrame with resolved feature names — and plays nicely with
``set_output(transform="pandas")``: it consumes DataFrames, exposes
``get_feature_names_out``, and returns DataFrames from ``transform``.
"""

from __future__ import annotations

import logging
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from ml_pipeline.core.types import TaskType

logger = logging.getLogger(__name__)


def _as_dataframe(X: pd.DataFrame | np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    """Coerce ``X`` to a DataFrame, generating ``x0..xN`` names when none exist.

    Args:
        X: input feature matrix (DataFrame passes through untouched).
        columns: column names to apply to a bare array; auto-generated if None.

    Returns:
        A pandas DataFrame view of ``X`` with string column names.
    """
    if isinstance(X, pd.DataFrame):
        return X
    array = np.asarray(X)
    if columns is None:
        columns = [f"x{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


class CorrelationPruner(TransformerMixin, BaseEstimator):
    """Drop the later feature of every pair whose |correlation| exceeds a threshold.

    Fit computes the absolute correlation matrix of the (numeric,
    post-preprocessing) DataFrame and greedily walks column pairs in order:
    whenever two still-kept columns correlate above ``threshold``, the
    later one is dropped. Transform selects the surviving columns.

    Attributes (post-fit):
        kept_columns_: ordered feature names that survived pruning.
        dropped_columns_: feature names that were pruned.
        feature_names_in_: input feature names seen during fit.
        n_features_in_: number of input features seen during fit.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        """Store the pruning threshold (sklearn convention: no validation here).

        Args:
            threshold: absolute-correlation cutoff in ``(0, 1]``; pairs strictly
                above it trigger a drop.
        """
        self.threshold = threshold

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> Self:
        """Compute correlations and decide which columns to keep.

        Args:
            X: all-numeric feature matrix (DataFrame expected; arrays get
                generated ``x0..xN`` names).
            y: ignored, present for pipeline compatibility.

        Returns:
            The fitted transformer.
        """
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]

        corr = X_df.corr().abs()
        columns = list(X_df.columns)
        dropped: set[str] = set()
        for i, col_i in enumerate(columns):
            if col_i in dropped:
                continue
            for col_j in columns[i + 1 :]:
                if col_j in dropped:
                    continue
                # NaN correlations (constant columns) compare False and are kept.
                if corr.loc[col_i, col_j] > self.threshold:
                    dropped.add(col_j)

        self.kept_columns_: list[str] = [c for c in columns if c not in dropped]
        self.dropped_columns_: list[str] = [c for c in columns if c in dropped]
        logger.info(
            "CorrelationPruner(threshold=%.3f): dropped %d/%d features %s",
            self.threshold,
            len(self.dropped_columns_),
            len(columns),
            self.dropped_columns_ or "(none)",
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Select the columns kept at fit time.

        Args:
            X: feature matrix with the same columns seen during fit.

        Returns:
            DataFrame restricted to ``kept_columns_``.
        """
        check_is_fitted(self, "kept_columns_")
        X_df = _as_dataframe(X, columns=list(self.feature_names_in_))
        return X_df.loc[:, self.kept_columns_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of the surviving output features.

        Args:
            input_features: ignored; kept for sklearn API compatibility.

        Returns:
            Array of kept column names.
        """
        check_is_fitted(self, "kept_columns_")
        return np.asarray(self.kept_columns_, dtype=object)


class ImportanceSelector(TransformerMixin, BaseEstimator):
    """Keep the features a quick random forest ranks as most important.

    Fit trains a small ``RandomForestClassifier``/``RandomForestRegressor``
    (task-dependent) on ``(X, y)``, ranks features by impurity importance, and
    keeps the ``top_k`` best — or, when ``top_k`` is None, every feature whose
    importance is at least the median. Transform selects the kept columns in
    their original order.

    Attributes (post-fit):
        selected_features_: kept feature names, in original column order.
        importances_: pd.Series of all input importances, sorted descending.
        feature_names_in_: input feature names seen during fit.
        n_features_in_: number of input features seen during fit.
    """

    def __init__(
        self,
        task: TaskType,
        top_k: int | None = None,
        n_estimators: int = 100,
        seed: int = 42,
    ) -> None:
        """Store selection settings (sklearn convention: no validation here).

        Args:
            task: classification or regression; picks the forest flavor.
            top_k: number of features to keep; None keeps those with
                importance >= median importance.
            n_estimators: trees in the ranking forest.
            seed: ``random_state`` for the ranking forest.
        """
        self.task = task
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """Rank features with a quick forest and record the kept subset.

        Args:
            X: all-numeric feature matrix.
            y: target values — required (supplied automatically when this
                transformer sits inside an sklearn ``Pipeline``).

        Returns:
            The fitted transformer.

        Raises:
            ValueError: if ``y`` is None.
        """
        if y is None:
            raise ValueError("ImportanceSelector requires y at fit time.")
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]

        forest_cls = (
            RandomForestClassifier
            if self.task == TaskType.CLASSIFICATION
            else RandomForestRegressor
        )
        forest = forest_cls(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        )
        forest.fit(X_df, np.asarray(y).ravel())

        importances = pd.Series(forest.feature_importances_, index=X_df.columns)
        if self.top_k is not None:
            k = min(self.top_k, len(importances))
            selected = set(importances.nlargest(k).index)
            rule = f"top_k={self.top_k}"
        else:
            selected = set(importances[importances >= importances.median()].index)
            rule = "importance >= median"

        self.selected_features_: list[str] = [c for c in X_df.columns if c in selected]
        self.importances_: pd.Series = importances.sort_values(ascending=False)
        logger.info(
            "ImportanceSelector (%s, %s): kept %d/%d features; top 5: %s",
            forest_cls.__name__,
            rule,
            len(self.selected_features_),
            len(importances),
            self.importances_.head(5).round(4).to_dict(),
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Select the features chosen at fit time.

        Args:
            X: feature matrix with the same columns seen during fit.

        Returns:
            DataFrame restricted to ``selected_features_``.
        """
        check_is_fitted(self, "selected_features_")
        X_df = _as_dataframe(X, columns=list(self.feature_names_in_))
        return X_df.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of the selected output features.

        Args:
            input_features: ignored; kept for sklearn API compatibility.

        Returns:
            Array of selected column names.
        """
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


class LoggingPCA(PCA):
    """A :class:`~sklearn.decomposition.PCA` that logs explained variance after fit.

    Behaves exactly like PCA, plus:
    - after every fit, logs the cumulative explained-variance curve;
    - output features are named ``pca_0..pca_{n-1}`` (stable, subclass-name-free),
      which flows through ``set_output(transform="pandas")`` as column names.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None) -> Self:
        """Fit PCA, then log the cumulative explained variance."""
        super().fit(X, y)
        self._log_explained_variance()
        return self

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: object = None) -> np.ndarray:
        """Fit PCA and transform ``X``, then log the cumulative explained variance."""
        transformed = super().fit_transform(X, y)
        self._log_explained_variance()
        return transformed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Output component names ``pca_0..pca_{n-1}``.

        Args:
            input_features: ignored; kept for sklearn API compatibility.

        Returns:
            Array of component names.
        """
        check_is_fitted(self, "n_components_")
        return np.asarray([f"pca_{i}" for i in range(self.n_components_)], dtype=object)

    def _log_explained_variance(self) -> None:
        """Log how much variance the fitted components retain."""
        cumulative = np.cumsum(self.explained_variance_ratio_)
        logger.info(
            "LoggingPCA: %d components retain %.4f of variance (n_components=%r)",
            self.n_components_,
            float(cumulative[-1]) if len(cumulative) else 0.0,
            self.n_components,
        )
        logger.debug(
            "Cumulative explained variance per component: %s",
            np.round(cumulative, 4).tolist(),
        )
