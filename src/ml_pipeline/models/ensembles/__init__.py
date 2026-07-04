"""Ensemble models built on the shared BaseModel contract (any registered model,
classical or deep, can be a base learner)."""

from ml_pipeline.models.ensembles import stacking, voting  # noqa: F401  (registration)
