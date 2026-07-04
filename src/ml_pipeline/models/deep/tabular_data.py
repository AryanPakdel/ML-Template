"""Dataset/DataLoader helpers for deep models trained on tabular feature matrices.

This module only covers the tabular case (post-preprocessing 2-D float matrices).
Other modalities plug in alongside it: an ``ImageDataset`` (decoding paths/bytes to
CHW tensors) or a ``SequenceDataset`` (padding/packing variable-length series)
would live in sibling modules of :mod:`ml_pipeline.models.deep`, expose the same
``(x, y)`` / ``x`` item contract, and be consumed by their model's ``fit`` via a
``build_dataloaders``-style factory — nothing outside this subpackage changes.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml_pipeline.core.types import TaskType

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """In-memory tabular dataset yielding float32 features and task-typed targets.

    Args:
        X: feature matrix, coerced to ``float32`` (``(n_samples, n_features)``).
        y: optional targets. Classification targets become ``torch.long`` (as
            expected by ``CrossEntropyLoss``); regression targets ``torch.float32``.
        task: decides the target dtype; ignored when ``y`` is ``None``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        task: TaskType = TaskType.CLASSIFICATION,
    ) -> None:
        self.X: torch.Tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if y is None:
            self.y: torch.Tensor | None = None
        else:
            target_dtype = torch.long if task == TaskType.CLASSIFICATION else torch.float32
            self.y = torch.as_tensor(np.asarray(y)).to(target_dtype)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Return ``(x, y)`` when targets exist, else ``x`` alone (inference mode)."""
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    task: TaskType,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders over :class:`TabularDataset` instances.

    Only the training loader shuffles; ``num_workers=0`` keeps behaviour portable
    (no fork/spawn differences across OSes) for these small in-memory tensors.

    Args:
        X_train: training features (``float32``-coercible).
        y_train: training targets.
        X_val: validation features used for early stopping.
        y_val: validation targets.
        batch_size: batch size shared by both loaders.
        task: forwarded to :class:`TabularDataset` for target dtype selection.

    Returns:
        ``(train_loader, val_loader)``.
    """
    train_loader = DataLoader(
        TabularDataset(X_train, y_train, task),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val, task),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader
