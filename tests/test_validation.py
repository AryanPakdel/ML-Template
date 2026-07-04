"""Pandera-backed schema validation driven by the declared column specs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.data.validation import DataValidationError, validate_dataframe


@pytest.fixture()
def clf_frame(synthetic_classification_df: pd.DataFrame) -> pd.DataFrame:
    """A mutable copy of the synthetic classification frame."""
    return synthetic_classification_df.copy()


def test_valid_frame_passes_and_coerces(
    clf_frame: pd.DataFrame, clf_config: PipelineConfig
) -> None:
    """A conforming frame validates; benign dtype drift is coerced back."""
    clf_frame["label"] = clf_frame["label"].astype("float64")  # drifted dtype
    validated = validate_dataframe(clf_frame, clf_config.data)
    assert isinstance(validated, pd.DataFrame)
    assert len(validated) == len(clf_frame)
    assert str(validated["label"].dtype) == "int64"
    assert str(validated["cat_a"].dtype) == "category"


def test_out_of_range_value_raises(clf_frame: pd.DataFrame, clf_config: PipelineConfig) -> None:
    """A value above the declared 'le' bound fails, naming the column."""
    clf_frame.loc[0, "num_a"] = 99.0
    with pytest.raises(DataValidationError, match="num_a"):
        validate_dataframe(clf_frame, clf_config.data)


def test_bad_category_raises(clf_frame: pd.DataFrame, clf_config: PipelineConfig) -> None:
    """A value outside allowed_values fails, naming the column."""
    clf_frame.loc[0, "cat_a"] = "purple"
    with pytest.raises(DataValidationError, match="cat_a"):
        validate_dataframe(clf_frame, clf_config.data)


def test_unexpected_null_raises(clf_frame: pd.DataFrame, clf_config: PipelineConfig) -> None:
    """A null in a non-nullable column fails, naming the column."""
    clf_frame.loc[0, "num_a"] = np.nan
    with pytest.raises(DataValidationError, match="num_a"):
        validate_dataframe(clf_frame, clf_config.data)
