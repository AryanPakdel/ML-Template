"""Pydantic v2 schema for the whole pipeline configuration.

Every dataset-specific fact (columns, dtypes, valid ranges, target name) lives in
config — ``src/`` stays dataset-agnostic. ``extra="forbid"`` everywhere so a typo
in a YAML key fails fast at load time instead of being silently ignored.

This module must not import any pipeline stage (models, preprocessing, ...);
string keys are resolved against the registries lazily by the orchestrator.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field, model_validator

from ml_pipeline.core.types import ColumnRole, TaskType


class StrictModel(PydanticModel):
    """Base for all config models: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class DataSourceConfig(StrictModel):
    """Where and how to load the raw dataset."""

    type: str = "csv"  # key into LOADER_REGISTRY ("csv", "parquet", ...)
    path: str
    read_options: dict[str, Any] = Field(default_factory=dict)


class ColumnSpec(StrictModel):
    """Declared schema for one raw column; feeds pandera validation, the
    preprocessing builder (column grouping), and the serving request model."""

    name: str
    dtype: Literal["int", "float", "category", "string", "bool", "datetime"]
    nullable: bool = False
    ge: float | None = None
    le: float | None = None
    allowed_values: list[Any] | None = None
    role: ColumnRole = ColumnRole.FEATURE


class SplitConfig(StrictModel):
    """Train/val/test split strategy."""

    strategy: Literal["random", "stratified", "time"] = "random"
    test_size: float = Field(0.15, gt=0, lt=1)
    val_size: float = Field(0.15, gt=0, lt=1)
    stratify_column: str | None = None  # defaults to the target at runtime
    time_column: str | None = None  # required for strategy="time"

    @model_validator(mode="after")
    def _check_sizes_and_time(self) -> SplitConfig:
        if self.test_size + self.val_size >= 0.9:
            raise ValueError("test_size + val_size must leave a real training set")
        if self.strategy == "time" and not self.time_column:
            raise ValueError("split.strategy='time' requires split.time_column")
        return self


class DataConfig(StrictModel):
    """Dataset definition: source, task, target, and full column schema."""

    source: DataSourceConfig
    target: str
    task: TaskType
    columns: list[ColumnSpec]
    split: SplitConfig = Field(default_factory=SplitConfig)

    @model_validator(mode="after")
    def _check_target_in_columns(self) -> DataConfig:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate column specs: {dupes}")
        if self.target not in names:
            raise ValueError(
                f"Target '{self.target}' missing from data.columns ({names})"
            )
        return self

    def feature_columns(self) -> list[ColumnSpec]:
        """Columns used as model inputs (role=feature, excluding the target)."""
        return [
            c
            for c in self.columns
            if c.role == ColumnRole.FEATURE and c.name != self.target
        ]

    def columns_by_role(self, role: ColumnRole) -> list[str]:
        """Names of columns with the given role."""
        return [c.name for c in self.columns if c.role == role]


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


class EdaConfig(StrictModel):
    """Automated EDA report settings; analyzers are keys into EDA_REGISTRY."""

    analyzers: list[str] = Field(
        default_factory=lambda: [
            "missing",
            "distributions",
            "categoricals",
            "correlation",
            "class_balance",
            "outliers",
            "cardinality",
            "leakage",
        ]
    )
    outlier_method: Literal["iqr", "zscore", "both"] = "iqr"
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    high_cardinality_threshold: int = 50
    leakage_corr_threshold: float = 0.98
    max_distribution_plots: int = 30


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


class OutlierConfig(StrictModel):
    """Outlier handling for numeric features.

    ``clip`` is applied inside the persisted pipeline; ``remove`` drops rows from
    the *training split only* (row-level ops never live in the inference path).
    """

    method: Literal["none", "clip", "remove"] = "none"
    iqr_factor: float = 1.5


class NumericPrepConfig(StrictModel):
    """Numeric column processing (keys into IMPUTER/SCALER registries)."""

    imputer: str = "median"
    imputer_options: dict[str, Any] = Field(default_factory=dict)
    scaler: str = "standard"
    scaler_options: dict[str, Any] = Field(default_factory=dict)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)


class CategoricalPrepConfig(StrictModel):
    """Categorical column processing (keys into IMPUTER/ENCODER registries)."""

    imputer: str = "most_frequent"
    imputer_options: dict[str, Any] = Field(default_factory=dict)
    encoder: str = "onehot"
    encoder_options: dict[str, Any] = Field(default_factory=dict)
    # Above this cardinality, one-hot silently degrades to frequency encoding
    # (with a logged warning) to avoid feature explosion.
    max_onehot_cardinality: int = 15


class DatetimePrepConfig(StrictModel):
    """Datetime feature extraction settings."""

    extract: list[
        Literal["year", "month", "day", "weekday", "hour", "is_weekend", "dayofyear"]
    ] = Field(default_factory=lambda: ["year", "month", "day", "weekday", "is_weekend"])


class ColumnOverride(StrictModel):
    """Per-column overrides of the global preprocessing choices."""

    imputer: str | None = None
    imputer_options: dict[str, Any] = Field(default_factory=dict)
    encoder: str | None = None
    encoder_options: dict[str, Any] = Field(default_factory=dict)
    scaler: str | None = None
    scaler_options: dict[str, Any] = Field(default_factory=dict)


class PreprocessingConfig(StrictModel):
    """Full preprocessing stage config, assembled into a ColumnTransformer."""

    numeric: NumericPrepConfig = Field(default_factory=NumericPrepConfig)
    categorical: CategoricalPrepConfig = Field(default_factory=CategoricalPrepConfig)
    datetime: DatetimePrepConfig = Field(default_factory=DatetimePrepConfig)
    column_overrides: dict[str, ColumnOverride] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# feature engineering / selection
# ---------------------------------------------------------------------------


class PolynomialConfig(StrictModel):
    """Config-gated polynomial/interaction feature generation."""

    enabled: bool = False
    degree: int = Field(2, ge=2, le=4)
    interaction_only: bool = True


class CorrelationPruningConfig(StrictModel):
    """Drop one of each pair of near-duplicate features."""

    enabled: bool = False
    threshold: float = Field(0.95, gt=0, le=1)


class ImportanceSelectionConfig(StrictModel):
    """Keep top features ranked by a quick baseline tree model."""

    enabled: bool = False
    top_k: int | None = None  # None -> use median importance threshold
    n_estimators: int = 100


class PcaConfig(StrictModel):
    """Optional dimensionality reduction with explained-variance logging."""

    enabled: bool = False
    n_components: float | int = 0.95  # float=variance kept, int=components


class FeatureConfig(StrictModel):
    """Feature engineering/selection stage (applied after preprocessing)."""

    polynomial: PolynomialConfig = Field(default_factory=PolynomialConfig)
    correlation_pruning: CorrelationPruningConfig = Field(
        default_factory=CorrelationPruningConfig
    )
    importance_selection: ImportanceSelectionConfig = Field(
        default_factory=ImportanceSelectionConfig
    )
    pca: PcaConfig = Field(default_factory=PcaConfig)


# ---------------------------------------------------------------------------
# model / training / tuning
# ---------------------------------------------------------------------------


class ImbalanceConfig(StrictModel):
    """Class imbalance handling (classification only).

    ``class_weight`` is passed to models that support it; samplers (keys into
    SAMPLER_REGISTRY: "smote", "random_over", "random_under") resample the
    *transformed training fold only* and are never part of the inference bundle.
    """

    strategy: Literal["none", "class_weight", "smote", "random_over", "random_under"] = (
        "none"
    )
    options: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictModel):
    """Which model to train and with what parameters.

    ``name`` is a key into MODEL_REGISTRY (validated lazily by the trainer so the
    config layer never imports model code). ``params`` are passed through to the
    model implementation untouched.
    """

    name: str = "random_forest"
    params: dict[str, Any] = Field(default_factory=dict)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)


class CvConfig(StrictModel):
    """Cross-validation strategy (keys into CV_SPLITTER_REGISTRY)."""

    strategy: Literal["kfold", "stratified", "timeseries"] = "kfold"
    n_splits: int = Field(5, ge=2)
    shuffle: bool = True


class TrainingConfig(StrictModel):
    """Training-stage settings beyond the model itself."""

    cv: CvConfig = Field(default_factory=CvConfig)


class SearchSpaceParam(StrictModel):
    """One hyperparameter dimension overriding the model's default search space."""

    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> SearchSpaceParam:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical search-space param needs 'choices'")
        elif self.low is None or self.high is None:
            raise ValueError(f"{self.type} search-space param needs 'low' and 'high'")
        return self


class TuningConfig(StrictModel):
    """Optuna study settings."""

    n_trials: int = 20
    timeout_s: int | None = None
    metric: str = "auto"  # "auto" -> f1 (classification) / rmse (regression)
    direction: Literal["auto", "maximize", "minimize"] = "auto"
    pruner: Literal["median", "none"] = "median"
    search_space: dict[str, SearchSpaceParam] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# evaluation / comparison
# ---------------------------------------------------------------------------


class ExplainabilityConfig(StrictModel):
    """SHAP/LIME settings for the best model."""

    method: Literal["auto", "shap", "lime", "none"] = "auto"
    max_samples: int = 200  # cap for KernelExplainer background/eval rows
    top_features: int = 10


class EvaluationConfig(StrictModel):
    """Metrics, explainability, and error-analysis settings."""

    metrics: list[str] | Literal["auto"] = "auto"
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    error_analysis_top_n: int = 20


class EnsembleConfig(StrictModel):
    """Ensembles built on top of the individually trained compare models."""

    voting: bool = False
    stacking: bool = False
    base_models: list[str] = Field(default_factory=list)


class CompareConfig(StrictModel):
    """Leaderboard run: train each listed model, then optional ensembles.

    ``model_params`` optionally overrides hyperparameters per listed model
    (e.g. a small ``mlp`` for smoke runs); unlisted models use their defaults.
    """

    models: list[str] = Field(default_factory=list)
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


# ---------------------------------------------------------------------------
# run / tracking / serving
# ---------------------------------------------------------------------------


class MlflowConfig(StrictModel):
    """MLflow tracking settings (local file store by default)."""

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"


class RunConfig(StrictModel):
    """Run-level settings: naming, seeding, artifact locations, logging."""

    experiment_name: str = "default"
    seed: int = 42
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    log_level: str = "INFO"
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)


class ServingConfig(StrictModel):
    """FastAPI serving settings; bundle_path may be 'latest' or an explicit path."""

    bundle_path: str = "latest"
    host: str = "0.0.0.0"
    port: int = 8000


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------


class PipelineConfig(StrictModel):
    """Root config: one validated object drives the entire pipeline."""

    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig
    eda: EdaConfig = Field(default_factory=EdaConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    @model_validator(mode="after")
    def _cross_field_rules(self) -> PipelineConfig:
        task = self.data.task
        split = self.data.split
        cv = self.training.cv
        imbalance = self.model.imbalance

        if task == TaskType.REGRESSION:
            if split.strategy == "stratified":
                raise ValueError("stratified split is only valid for classification")
            if cv.strategy == "stratified":
                raise ValueError("stratified CV is only valid for classification")
            if imbalance.strategy != "none":
                raise ValueError("imbalance handling is only valid for classification")

        if split.strategy == "time":
            if cv.strategy != "timeseries":
                raise ValueError("time-based split requires training.cv.strategy='timeseries'")
            if cv.shuffle:
                raise ValueError("time-based split requires training.cv.shuffle=false")
            if imbalance.strategy not in ("none", "class_weight"):
                raise ValueError("resampling (SMOTE & co.) breaks temporal order; use 'none' or 'class_weight'")

        if cv.strategy == "timeseries" and cv.shuffle:
            raise ValueError("timeseries CV cannot shuffle")

        return self
