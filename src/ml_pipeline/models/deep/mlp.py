"""Multi-layer perceptron for tabular data (PyTorch + Lightning) behind BaseModel.

The Lightning machinery (trainer, early stopping, checkpointing) is an internal
implementation detail: from the outside this is just another registry model with
``fit``/``predict``/``predict_proba``, picklable inside a ``PipelineBundle``.
"""

from __future__ import annotations

import io
import logging
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Self

import lightning
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn

from ml_pipeline.core.types import ExplainerHint, TaskType
from ml_pipeline.models.base import MODEL_REGISTRY, ArrayLike, BaseModel
from ml_pipeline.models.deep.tabular_data import build_dataloaders

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# Fraction of the training split carved out for early stopping when no explicit
# validation set is provided.
_VAL_FRACTION = 0.1

# Layer widths sampled by the default Optuna search space.
_WIDTH_CHOICES: tuple[int, ...] = (32, 64, 128, 256)

# Hyperparameter defaults; every key is overridable via ``params`` (YAML/Optuna).
# "callbacks" is a private hook for extra Lightning callbacks injected by the
# tuner (e.g. pruning) and is never pickled.
_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [128, 64],
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 10,
    "weight_decay": 1e-4,
    "gradient_clip_val": 1.0,
    "mixed_precision": False,
    "callbacks": [],
}


def _as_float32(X: ArrayLike) -> np.ndarray:
    """Coerce a DataFrame or array-like feature matrix to a float32 numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _carve_holdout(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split ``(X, y)`` into train/holdout for early stopping.

    Returns:
        ``(X_train, y_train, X_val, y_val)`` with ``_VAL_FRACTION`` of the rows
        (at least one) in the holdout, shuffled by a seeded generator.
    """
    n_samples = len(X)
    n_val = max(1, int(round(n_samples * _VAL_FRACTION)))
    permutation = np.random.default_rng(seed).permutation(n_samples)
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class _TabularMLP(lightning.LightningModule):
    """Private LightningModule: Linear/ReLU/Dropout stack with task-aware loss."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        task: TaskType,
    ) -> None:
        """Build the network; ``output_dim`` is ``n_classes`` or 1 (regression)."""
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev_dim, width), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.task = task
        self.loss_fn: nn.Module = (
            nn.CrossEntropyLoss() if task == TaskType.CLASSIFICATION else nn.MSELoss()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw network outputs: logits (classification) or values (regression)."""
        return self.network(x)

    def _step_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Shared train/val loss computation."""
        x, y = batch
        outputs = self(x)
        if self.task == TaskType.REGRESSION:
            outputs = outputs.squeeze(-1)
        return self.loss_fn(outputs, y)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One optimisation step; logs ``train_loss``."""
        loss = self._step_loss(batch)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One validation step; logs ``val_loss`` (monitored by callbacks)."""
        loss = self._step_loss(batch)
        self.log("val_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW plus ReduceLROnPlateau stepping on the monitored ``val_loss``."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


@MODEL_REGISTRY.register("mlp")
class MLPModel(BaseModel):
    """Feed-forward neural network for tabular classification and regression.

    Trains with early stopping on ``val_loss`` (carving a seeded 10% holdout when
    no validation split is supplied), restores the best checkpoint, and pickles
    its weights as bytes so it survives ``joblib`` round-trips like any other
    model in the registry.
    """

    name: ClassVar[str] = "mlp"
    supported_tasks: ClassVar[frozenset[TaskType]] = frozenset(
        {TaskType.CLASSIFICATION, TaskType.REGRESSION}
    )
    explainer_hint: ClassVar[ExplainerHint] = ExplainerHint.KERNEL

    def __init__(self, params: dict[str, Any], task: TaskType, seed: int = 42) -> None:
        """Store hyperparameters; the network is built lazily in :meth:`fit`."""
        super().__init__(params, task, seed)
        self._network: _TabularMLP | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None

    # ------------------------------------------------------------- internals
    def _hp(self, key: str) -> Any:
        """Hyperparameter lookup with module-level defaults."""
        return self.params.get(key, _DEFAULTS[key])

    def _build_network(self, input_dim: int, output_dim: int) -> _TabularMLP:
        """Construct a fresh ``_TabularMLP`` from the current hyperparameters."""
        return _TabularMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[int(h) for h in self._hp("hidden_dims")],
            dropout=float(self._hp("dropout")),
            lr=float(self._hp("lr")),
            weight_decay=float(self._hp("weight_decay")),
            task=self.task,
        )

    def _forward_batched(self, X: ArrayLike) -> torch.Tensor:
        """Batched no-grad forward pass on CPU; returns stacked raw outputs."""
        if self._network is None:
            raise RuntimeError(f"Model '{self.name}' is not fitted yet; call fit() first.")
        features = torch.as_tensor(_as_float32(X))
        batch_size = int(self._hp("batch_size"))
        network = self._network.cpu().eval()
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                outputs.append(network(features[start : start + batch_size]))
        if not outputs:
            return torch.empty((0, self._output_dim or 1))
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------ api
    def fit(
        self,
        X: ArrayLike,
        y: np.ndarray,
        X_val: ArrayLike | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Train with early stopping and restore the best-``val_loss`` weights.

        For classification ``y`` arrives label-encoded (``0..n_classes-1``), so
        ``n_classes`` is inferred as ``int(y.max()) + 1``.
        """
        self._remember_features(X)
        if sample_weight is not None:
            logger.warning("Model '%s' does not support sample_weight; ignoring it.", self.name)

        X_arr = _as_float32(X)
        y_arr = np.asarray(y)

        if self.task == TaskType.CLASSIFICATION:
            self.n_classes_ = int(y_arr.max()) + 1
            output_dim = self.n_classes_
        else:
            output_dim = 1

        if X_val is not None and y_val is not None:
            X_tr, y_tr = X_arr, y_arr
            X_va, y_va = _as_float32(X_val), np.asarray(y_val)
        else:
            logger.debug(
                "No validation split provided; carving %.0f%% of train for early stopping.",
                _VAL_FRACTION * 100,
            )
            X_tr, y_tr, X_va, y_va = _carve_holdout(X_arr, y_arr, self.seed)

        lightning.seed_everything(self.seed)
        train_loader, val_loader = build_dataloaders(
            X_tr, y_tr, X_va, y_va, int(self._hp("batch_size")), self.task
        )

        self._input_dim = int(X_arr.shape[1])
        self._output_dim = output_dim
        network = self._build_network(self._input_dim, self._output_dim)

        checkpoint_dir = tempfile.mkdtemp(prefix="mlp_ckpt_")
        checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=checkpoint_dir, save_top_k=1)
        callbacks: list[Any] = [
            EarlyStopping(monitor="val_loss", patience=int(self._hp("patience"))),
            checkpoint,
            *list(self._hp("callbacks")),
        ]
        trainer = lightning.Trainer(
            max_epochs=int(self._hp("max_epochs")),
            accelerator="auto",
            devices=1,
            gradient_clip_val=float(self._hp("gradient_clip_val")),
            precision="16-mixed" if bool(self._hp("mixed_precision")) else "32-true",
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        try:
            trainer.fit(network, train_loader, val_loader)
            best_path = checkpoint.best_model_path
            if best_path:
                # Our own just-written Lightning checkpoint; contains optimizer
                # state etc., hence weights_only=False.
                state = torch.load(best_path, map_location="cpu", weights_only=False)
                network.load_state_dict(state["state_dict"])
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        self._network = network.cpu().eval()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Argmax class indices (classification) or squeezed values (regression)."""
        outputs = self._forward_batched(X)
        if self.task == TaskType.CLASSIFICATION:
            return outputs.argmax(dim=1).numpy()
        return outputs.squeeze(-1).numpy()

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Softmax class probabilities; regression defers to the base (raises)."""
        if self.task != TaskType.CLASSIFICATION:
            return super().predict_proba(X)
        return torch.softmax(self._forward_batched(X), dim=1).numpy()

    # -------------------------------------------------------------- pickling
    def __getstate__(self) -> dict[str, Any]:
        """Serialize weights to bytes; drop unpicklable injected callbacks."""
        weights: bytes | None = None
        if self._network is not None:
            buffer = io.BytesIO()
            torch.save(self._network.state_dict(), buffer)
            weights = buffer.getvalue()
        return {
            "params": {k: v for k, v in self.params.items() if k != "callbacks"},
            "task": self.task,
            "seed": self.seed,
            "feature_names_": self.feature_names_,
            "n_classes_": self.n_classes_,
            "_input_dim": self._input_dim,
            "_output_dim": self._output_dim,
            "weights": weights,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild the network and load weights; tolerate the unfitted case."""
        self.params = state["params"]
        self.task = state["task"]
        self.seed = state["seed"]
        self.feature_names_ = state["feature_names_"]
        self.n_classes_ = state["n_classes_"]
        self._input_dim = state["_input_dim"]
        self._output_dim = state["_output_dim"]
        self._network = None
        weights = state.get("weights")
        if weights is not None and self._input_dim is not None and self._output_dim is not None:
            self._network = self._build_network(self._input_dim, self._output_dim)
            self._network.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
            self._network.eval()

    # ---------------------------------------------------------------- tuning
    @classmethod
    def get_default_search_space(cls, trial: optuna.Trial, task: TaskType) -> dict[str, Any]:
        """Default Optuna space: depth/width, dropout, lr, batch size, weight decay."""
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = [
            trial.suggest_categorical(f"hidden_dim_{i}", list(_WIDTH_CHOICES))
            for i in range(n_layers)
        ]
        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
