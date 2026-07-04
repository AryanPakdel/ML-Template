"""Classical (scikit-learn-family) model implementations.

Importing this subpackage registers every classical model in ``MODEL_REGISTRY``
as a side effect. Adding a new classical model = one new module here + one
import line below.
"""

from ml_pipeline.models.classical import (
    catboost_model,  # noqa: F401  (registration)
    knn,  # noqa: F401  (registration)
    lightgbm_model,  # noqa: F401  (registration)
    linear,  # noqa: F401  (registration)
    naive_bayes,  # noqa: F401  (registration)
    random_forest,  # noqa: F401  (registration)
    svm,  # noqa: F401  (registration)
    tree,  # noqa: F401  (registration)
    xgboost_model,  # noqa: F401  (registration)
)
