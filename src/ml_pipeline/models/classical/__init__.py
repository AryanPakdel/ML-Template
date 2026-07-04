"""Classical (scikit-learn-family) model implementations.

Importing this subpackage registers every classical model in ``MODEL_REGISTRY``
as a side effect. Adding a new classical model = one new module here + one
import line below.
"""

from ml_pipeline.models.classical import catboost_model  # noqa: F401  (registration)
from ml_pipeline.models.classical import knn  # noqa: F401  (registration)
from ml_pipeline.models.classical import lightgbm_model  # noqa: F401  (registration)
from ml_pipeline.models.classical import linear  # noqa: F401  (registration)
from ml_pipeline.models.classical import naive_bayes  # noqa: F401  (registration)
from ml_pipeline.models.classical import random_forest  # noqa: F401  (registration)
from ml_pipeline.models.classical import svm  # noqa: F401  (registration)
from ml_pipeline.models.classical import tree  # noqa: F401  (registration)
from ml_pipeline.models.classical import xgboost_model  # noqa: F401  (registration)
