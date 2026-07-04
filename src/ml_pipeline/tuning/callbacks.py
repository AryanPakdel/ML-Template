"""Custom Optuna pruning callback for PyTorch Lightning.

Deliberately hand-rolled (~20 lines) instead of depending on
``optuna-integration``, whose Lightning coupling churns between releases.
"""

from __future__ import annotations

import lightning
import optuna


class OptunaPruningCallback(lightning.Callback):
    """Report ``val_loss`` to an Optuna trial each validation epoch and prune.

    The internal step counter is monotonic across CV folds (each fold re-uses the
    same trial), which keeps ``trial.report`` steps strictly increasing as Optuna
    requires.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss") -> None:
        self._trial = trial
        self._monitor = monitor
        self._step = 0

    def on_validation_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        """Report the monitored metric; raise ``TrialPruned`` when Optuna says stop."""
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self._monitor)
        if value is None:
            return
        self._trial.report(float(value), step=self._step)
        self._step += 1
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step {self._step} ({self._monitor}={value:.5f})")
