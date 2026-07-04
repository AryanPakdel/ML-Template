"""Unregistered skeletons for future deep architectures (CNN, LSTM, Transformer).

These classes subclass :class:`~ml_pipeline.models.base.BaseModel` so the exact
signatures a real implementation must satisfy are visible, but they are *not*
decorated with ``@MODEL_REGISTRY.register`` — they never appear in configs or the
CLI until someone fills them in. :mod:`ml_pipeline.models.deep.mlp` is the
reference implementation to copy: Lightning training loop, temp-dir
checkpointing, byte-serialized weights for pickling, and an Optuna search space.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import ArrayLike, BaseModel

_STUB_MESSAGE = (
    "'{name}' is a skeleton. See its class docstring for the three pieces "
    "(Dataset, network, registration) needed to make it a real model."
)


class CnnModel(BaseModel):
    """Skeleton for a convolutional network over image inputs.

    To plug this in:

    1. **Dataset** — add an ``ImageDataset`` module next to
       :mod:`ml_pipeline.models.deep.tabular_data` that maps rows (image paths or
       raw bytes coming out of the data stage) to normalized CHW ``float32``
       tensors plus task-typed targets, and a ``build_dataloaders`` factory
       mirroring the tabular one (shuffle train only, ``num_workers`` from
       params, standard augmentation transforms behind config flags).
    2. **Network** — implement a private ``_ImageCNN(lightning.LightningModule)``
       (conv/pool blocks -> global pooling -> classifier/regressor head) with
       ``training_step``/``validation_step`` logging ``train_loss``/``val_loss``
       and an AdamW + ReduceLROnPlateau ``configure_optimizers``; then write
       ``fit``/``predict``/``predict_proba`` and ``__getstate__``/``__setstate__``
       exactly like :class:`~ml_pipeline.models.deep.mlp.MLPModel` (early
       stopping, best-checkpoint restore, weights pickled as bytes).
    3. **Register** — add ``@MODEL_REGISTRY.register("cnn")`` above this class
       and import the module in ``ml_pipeline/models/deep/__init__.py`` (the
       ``stubs`` import already covers this file; move the class to its own
       module if it grows).
    """

    name: ClassVar[str] = "cnn"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.NONE

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Not implemented — see the class docstring for the plug-in recipe."""
        raise NotImplementedError(_STUB_MESSAGE.format(name=self.name))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Not implemented — see the class docstring for the plug-in recipe."""
        raise NotImplementedError(_STUB_MESSAGE.format(name=self.name))


class LstmModel(BaseModel):
    """Skeleton for a recurrent (LSTM/GRU) network over sequential inputs.

    To plug this in:

    1. **Dataset** — add a ``SequenceDataset`` module next to
       :mod:`ml_pipeline.models.deep.tabular_data` that windows/groups rows into
       ``(seq_len, n_features)`` ``float32`` tensors (grouping and ordering
       columns come from config, never hardcoded), handles variable lengths via
       padding + lengths (for ``pack_padded_sequence``) through a custom
       ``collate_fn``, and exposes a ``build_dataloaders`` factory.
    2. **Network** — implement a private ``_SequenceLSTM(lightning.LightningModule)``
       (``nn.LSTM`` -> last-hidden-state or attention pooling -> linear head)
       logging ``train_loss``/``val_loss``; reuse the ``fit`` shape of
       :class:`~ml_pipeline.models.deep.mlp.MLPModel` (seeded holdout when no
       val split, EarlyStopping + ModelCheckpoint in a temp dir, best-weight
       restore) and its byte-based ``__getstate__``/``__setstate__``.
    3. **Register** — add ``@MODEL_REGISTRY.register("lstm")`` above this class
       and ensure its module is imported from
       ``ml_pipeline/models/deep/__init__.py``.
    """

    name: ClassVar[str] = "lstm"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.NONE

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Not implemented — see the class docstring for the plug-in recipe."""
        raise NotImplementedError(_STUB_MESSAGE.format(name=self.name))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Not implemented — see the class docstring for the plug-in recipe."""
        raise NotImplementedError(_STUB_MESSAGE.format(name=self.name))


class TransformerModel(BaseModel):
    """Skeleton for a Transformer encoder (sequences or FT-Transformer-style tabular).

    To plug this in:

    1. **Dataset** — reuse the ``SequenceDataset`` described on
       :class:`LstmModel` for sequential inputs, or for tabular attention
       (FT-Transformer) add a dataset that keeps numeric/categorical columns
       separate so each feature can be embedded as a token (column grouping
       driven by config, mirroring :mod:`ml_pipeline.models.deep.tabular_data`).
    2. **Network** — implement a private
       ``_TransformerNet(lightning.LightningModule)`` (feature/positional
       embeddings -> ``nn.TransformerEncoder`` -> CLS-token or mean pooling ->
       linear head) logging ``train_loss``/``val_loss``; warmup + cosine or
       ReduceLROnPlateau scheduling via ``configure_optimizers``; copy the
       ``fit``/pickling structure of
       :class:`~ml_pipeline.models.deep.mlp.MLPModel` and add hyperparameters
       (``n_heads``, ``n_blocks``, ``d_model``, ...) to
       ``get_default_search_space``.
    3. **Register** — add ``@MODEL_REGISTRY.register("transformer")`` above this
       class and ensure its module is imported from
       ``ml_pipeline/models/deep/__init__.py``.
    """

    name: ClassVar[str] = "transformer"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.NONE

    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Not implemented — see the class docstring for the plug-in recipe."""
        raise NotImplementedError(_STUB_MESSAGE.format(name=self.name))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Not implemented — see the class docstring for the plug-in recipe."""
        raise NotImplementedError(_STUB_MESSAGE.format(name=self.name))


__all__ = ["CnnModel", "LstmModel", "TransformerModel"]
