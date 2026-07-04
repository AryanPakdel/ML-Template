"""The persisted training artifact and the single training-to-inference seam.

A run produces one directory under ``artifacts/runs/<run_id>/`` whose key file is
``bundle.joblib`` — a pickled :class:`PipelineBundle` holding the fitted
preprocessor, feature pipeline, model, and metadata. Serving and batch inference
load exactly this one file, so preprocessing and model versions can never diverge.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.core.types import TaskType
from ml_pipeline.utils.io import ensure_dir, write_json

BUNDLE_FILENAME = "bundle.joblib"
RUNS_SUBDIR = "runs"


@dataclass
class BundleMetadata:
    """Everything inference needs to know about how the bundle was trained."""

    run_id: str
    created_at: str
    experiment_name: str
    task: str  # TaskType value
    target: str
    model_name: str
    feature_columns: list[str]  # ordered raw input columns the bundle expects
    raw_feature_schema: list[dict[str, Any]]  # ColumnSpec dumps for those columns
    class_labels: list[Any] | None = None  # original labels, index = encoded class
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    package_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for metadata.json."""
        return asdict(self)


def new_run_id(experiment_name: str, model_name: str) -> str:
    """Timestamped run id, unique enough for a local file store."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{experiment_name}_{model_name}"


def create_run_dir(artifacts_dir: str | Path, run_id: str) -> Path:
    """Create ``artifacts/runs/<run_id>/`` (suffixing on collision) and return it."""
    base = Path(artifacts_dir) / RUNS_SUBDIR
    run_dir = base / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{run_id}-{suffix}"
        suffix += 1
    return ensure_dir(run_dir)


def resolve_bundle_path(spec: str | Path, artifacts_dir: str | Path = "artifacts") -> Path:
    """Resolve a bundle spec to a concrete ``bundle.joblib`` path.

    Args:
        spec: ``"latest"`` (newest bundle under ``artifacts/runs``), a run
            directory, or a direct path to a ``bundle.joblib``.
        artifacts_dir: root artifacts directory used for ``"latest"``.

    Raises:
        FileNotFoundError: when nothing matches, with a hint to train first.
    """
    if str(spec) == "latest":
        candidates = sorted(
            Path(artifacts_dir).glob(f"{RUNS_SUBDIR}/*/{BUNDLE_FILENAME}"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No bundles found under {Path(artifacts_dir) / RUNS_SUBDIR}. "
                "Train a model first (mlpipe train --config ...)."
            )
        return candidates[-1]

    path = Path(spec)
    if path.is_dir():
        path = path / BUNDLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")
    return path


class PipelineBundle:
    """Fitted preprocessor + feature pipeline + model + metadata, as one unit.

    ``predict``/``predict_proba`` accept a **raw** DataFrame with the original
    dataset's feature columns; all transformation happens inside, guaranteeing
    zero train/serve skew.
    """

    def __init__(
        self,
        preprocessor: Any,
        feature_pipeline: Any | None,
        model: Any,
        metadata: BundleMetadata,
    ) -> None:
        self.preprocessor = preprocessor
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.metadata = metadata

    # ------------------------------------------------------------------ io
    def save(self, run_dir: str | Path) -> Path:
        """Persist the bundle (and a human-readable metadata.json) into ``run_dir``."""
        run_dir = ensure_dir(Path(run_dir))
        bundle_path = run_dir / BUNDLE_FILENAME
        joblib.dump(self, bundle_path)
        write_json(self.metadata.to_dict(), run_dir / "metadata.json")
        return bundle_path

    @classmethod
    def load(cls, path: str | Path) -> PipelineBundle:
        """Load a bundle from a run directory or a direct bundle.joblib path."""
        path = Path(path)
        if path.is_dir():
            path = path / BUNDLE_FILENAME
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} is not a PipelineBundle (got {type(bundle).__name__})")
        return bundle

    # ------------------------------------------------------------- inference
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subset/reorder raw input to the training feature columns, loudly."""
        missing = [c for c in self.metadata.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input is missing required feature columns: {missing}. "
                f"Expected: {self.metadata.feature_columns}"
            )
        return df[self.metadata.feature_columns]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the exact fitted preprocessing + feature pipeline used in training."""
        features = self._select_features(df)
        transformed = self.preprocessor.transform(features)
        if self.feature_pipeline is not None:
            transformed = self.feature_pipeline.transform(transformed)
        return transformed

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict from raw input; classification outputs original class labels."""
        predictions = self.model.predict(self.transform(df))
        if (
            self.metadata.task == TaskType.CLASSIFICATION.value
            and self.metadata.class_labels is not None
        ):
            labels = np.asarray(self.metadata.class_labels)
            return labels[predictions.astype(int)]
        return predictions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Class probabilities from raw input (classification models only)."""
        return self.model.predict_proba(self.transform(df))
