"""Custom sklearn-compatible transformers used by the preprocessing builder.

All transformers here follow the scikit-learn estimator protocol
(``BaseEstimator`` + ``TransformerMixin``), operate on pandas DataFrames,
return pandas DataFrames, and implement ``get_feature_names_out`` so they
compose cleanly inside ``ColumnTransformer`` with
``set_output(transform="pandas")``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

#: Datetime parts supported by :class:`DatetimeFeatureExtractor`.
SUPPORTED_DATETIME_PARTS: frozenset[str] = frozenset(
    {"year", "month", "day", "weekday", "hour", "is_weekend", "dayofyear"}
)


def _require_dataframe(X: object, transformer_name: str) -> pd.DataFrame:
    """Return ``X`` if it is a DataFrame, otherwise raise a clear ``TypeError``.

    Args:
        X: candidate input.
        transformer_name: name used in the error message.

    Raises:
        TypeError: when ``X`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{transformer_name} expects a pandas DataFrame, got {type(X).__name__}. "
            "Ensure upstream steps use set_output(transform='pandas')."
        )
    return X


def _check_columns(X: pd.DataFrame, expected: Iterable[str], transformer_name: str) -> None:
    """Raise ``ValueError`` when any fitted column is missing from ``X``."""
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(
            f"{transformer_name}.transform input is missing fitted columns: {missing}"
        )


class DatetimeFeatureExtractor(TransformerMixin, BaseEstimator):
    """Expand datetime columns into numeric calendar parts.

    Each input column is coerced with ``pd.to_datetime(errors="coerce")`` and
    replaced by one ``<col>_<part>`` column per requested part. Unparseable
    values become ``NaT`` and yield ``NaN`` parts (``is_weekend`` yields 0),
    which downstream imputers handle. The original columns are dropped.

    Args:
        parts: parts to extract; each must be one of
            ``{"year", "month", "day", "weekday", "hour", "is_weekend", "dayofyear"}``.
    """

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DatetimeFeatureExtractor:
        """Validate ``parts`` and record the input columns.

        Raises:
            ValueError: when ``parts`` is empty or contains unsupported names.
        """
        X = _require_dataframe(X, type(self).__name__)
        if not self.parts:
            raise ValueError("DatetimeFeatureExtractor requires at least one part to extract.")
        invalid = sorted(set(self.parts) - SUPPORTED_DATETIME_PARTS)
        if invalid:
            raise ValueError(
                f"Unsupported datetime parts {invalid}. "
                f"Supported: {sorted(SUPPORTED_DATETIME_PARTS)}"
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with one ``<col>_<part>`` column per fitted column/part."""
        check_is_fitted(self, "feature_names_in_")
        X = _require_dataframe(X, type(self).__name__)
        _check_columns(X, self.feature_names_in_, type(self).__name__)

        out: dict[str, pd.Series] = {}
        for col in self.feature_names_in_:
            series = pd.to_datetime(X[col], errors="coerce")
            for part in self.parts:
                name = f"{col}_{part}"
                if part == "is_weekend":
                    out[name] = (series.dt.weekday >= 5).astype(int)
                else:
                    out[name] = getattr(series.dt, part)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Generated ``<col>_<part>`` names, in transform output order."""
        check_is_fitted(self, "feature_names_in_")
        columns = self.feature_names_in_ if input_features is None else input_features
        names = [f"{col}_{part}" for col in columns for part in self.parts]
        return np.asarray(names, dtype=object)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical values by their relative frequency on the fit data.

    ``fit`` stores a per-column ``value -> relative frequency`` map (normalized
    value counts, NaN excluded). ``transform`` maps values through it; unseen
    values and missing values map to ``0.0``. Output keeps the input column
    names with float dtype.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FrequencyEncoder:
        """Learn per-column relative frequency maps from the fit data."""
        X = _require_dataframe(X, type(self).__name__)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.frequency_maps_: dict[str, dict[object, float]] = {
            col: X[col].value_counts(normalize=True, dropna=True).to_dict()
            for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map values to fitted frequencies; unseen/NaN become ``0.0``."""
        check_is_fitted(self, "frequency_maps_")
        X = _require_dataframe(X, type(self).__name__)
        _check_columns(X, self.feature_names_in_, type(self).__name__)

        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            mapped = X[col].map(self.frequency_maps_[col])
            out[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0).astype(float)
        return out

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """One-to-one output names (same as the fitted input columns)."""
        check_is_fitted(self, "frequency_maps_")
        columns = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(columns, dtype=object)


class OutlierClipper(TransformerMixin, BaseEstimator):
    """Clip numeric columns to IQR-based bounds learned on the fit data.

    ``fit`` computes per-column ``[q1 - f*iqr, q3 + f*iqr]``; ``transform``
    clips values into that interval. Column names pass through one-to-one.

    Args:
        iqr_factor: multiplier ``f`` applied to the interquartile range.
    """

    def __init__(self, iqr_factor: float = 1.5) -> None:
        self.iqr_factor = iqr_factor

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> OutlierClipper:
        """Compute per-column clipping bounds from quartiles of the fit data."""
        X = _require_dataframe(X, type(self).__name__)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_: pd.Series = q1 - self.iqr_factor * iqr
        self.upper_bounds_: pd.Series = q3 + self.iqr_factor * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip each fitted column into its learned ``[lower, upper]`` interval."""
        check_is_fitted(self, "lower_bounds_")
        X = _require_dataframe(X, type(self).__name__)
        _check_columns(X, self.feature_names_in_, type(self).__name__)
        subset = X[list(self.feature_names_in_)]
        return subset.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """One-to-one output names (same as the fitted input columns)."""
        check_is_fitted(self, "lower_bounds_")
        columns = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(columns, dtype=object)
