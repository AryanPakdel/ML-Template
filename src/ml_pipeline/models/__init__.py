"""Model implementations behind one uniform :class:`~ml_pipeline.models.base.BaseModel`
contract, discovered via ``MODEL_REGISTRY``.

Importing this package triggers registration of every built-in model (classical,
deep learning, and ensembles). Adding a new model = one new file + one import in
the relevant subpackage ``__init__``.
"""

from ml_pipeline.models import classical, deep, ensembles  # noqa: F401  (registration)
from ml_pipeline.models.base import MODEL_REGISTRY, BaseModel

__all__ = ["MODEL_REGISTRY", "BaseModel"]
