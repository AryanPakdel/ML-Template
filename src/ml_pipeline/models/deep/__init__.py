"""Deep-learning models (PyTorch + Lightning) satisfying the BaseModel contract.

Importing this subpackage registers the built-in deep models (currently ``mlp``)
in ``MODEL_REGISTRY``. :mod:`~ml_pipeline.models.deep.tabular_data` holds the
shared Dataset/DataLoader helpers, and :mod:`~ml_pipeline.models.deep.stubs`
documents how to plug in CNN/LSTM/Transformer architectures.
"""

from ml_pipeline.models.deep import mlp, stubs, tabular_data  # noqa: F401  (registration)
