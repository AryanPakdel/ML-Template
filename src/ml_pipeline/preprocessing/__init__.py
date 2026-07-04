"""Preprocessing stage: config-driven sklearn ColumnTransformer construction.

``build_preprocessor`` turns :class:`~ml_pipeline.config.schema.PreprocessingConfig`
plus the dataset's declared column schema into an unfitted sklearn ``Pipeline``
(imputation, outlier clipping, scaling, encoding, datetime expansion). The
fitted result is persisted inside the
:class:`~ml_pipeline.core.artifacts.PipelineBundle`, so training and inference
share identical transformations.

Importing this package populates the component registries
(``IMPUTER_REGISTRY``, ``SCALER_REGISTRY``, ``ENCODER_REGISTRY``).
"""

from __future__ import annotations

from ml_pipeline.preprocessing import builder, components, transformers
from ml_pipeline.preprocessing.builder import build_preprocessor

__all__ = ["build_preprocessor", "builder", "components", "transformers"]
