"""FastAPI serving layer.

Everything here is derived from the persisted :class:`~ml_pipeline.core.artifacts.
PipelineBundle` metadata: the request model, value-constraint checks, and the
endpoints. No dataset column name is hardcoded, and no training/data stage is
imported — serving stays self-contained on the bundle.
"""

from __future__ import annotations
