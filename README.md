# ML Pipeline Template

A reusable, config-driven, end-to-end machine learning pipeline template for **tabular
classification and regression**. Every stage — data loading, validation, EDA, preprocessing,
feature engineering, model choice, cross-validation, tuning, evaluation, explainability, and
serving — is selected by YAML config through a small set of **registries**, so swapping a dataset
or a model is a config edit, not a code change. `src/` contains zero dataset-specific code: column
names, dtypes, and valid ranges live entirely in `configs/data/*.yaml`, and the serving API derives
its request schema from the trained bundle's metadata. Train once, get a single `bundle.joblib`
(preprocessor + feature pipeline + model + metadata), and serve it with zero train/serve skew.

## Features

- **Schema-validated ingestion** — the declared column schema is compiled into a
  [pandera](https://pandera.readthedocs.io/) `DataFrameSchema`; bad data fails fast with one
  readable error.
- **Automated EDA** — a self-contained HTML report (missing values, distributions, categoricals,
  correlation, class balance, outliers, cardinality, target-leakage screening), each section a
  pluggable analyzer.
- **Leakage-safe preprocessing** — imputation/scaling/encoding assembled into a sklearn
  `ColumnTransformer` fitted on training folds only and persisted inside the serving bundle;
  per-column overrides; automatic one-hot → frequency-encoding fallback for high-cardinality
  columns.
- **Feature engineering** — config-gated polynomial/interaction features, correlation pruning,
  importance-based selection, and PCA.
- **11 models behind one interface** — logistic/linear regression, decision tree, random forest,
  k-NN, SVM, naive Bayes, XGBoost, LightGBM, CatBoost, and a PyTorch Lightning MLP — plus
  **voting and stacking ensembles**. All share one `BaseModel` contract and are discovered via
  `MODEL_REGISTRY`.
- **Per-fold cross-validation** — k-fold / stratified / time-series splitters with per-fold
  preprocessing to prevent leakage.
- **Imbalance handling** — class weighting or SMOTE / random over/under-sampling (applied to
  training folds only, never part of the inference path).
- **Optuna tuning** — per-model default search spaces, YAML-overridable, median pruning, then an
  automatic re-fit + persist of the best model.
- **MLflow tracking** — params, per-fold and final metrics, and artifacts logged to a local
  SQLite store (`mlflow.db`) by default.
- **Explainability** — SHAP with automatic explainer dispatch (tree/linear/kernel) and a LIME
  fallback; explainability never fails a run.
- **Evaluation extras** — per-task metric suites, ROC/PR/confusion-matrix plots, feature
  importances, and a worst-errors CSV for error analysis.
- **FastAPI serving** — `mlpipe serve` loads a bundle and exposes prediction endpoints whose
  request model is derived from bundle metadata.
- **Docker, CI, pytest** — containerized serving via docker compose, GitHub Actions workflow, and
  a pytest suite.

## Setup

Python 3.11+. The primary path is [miniforge/mamba](https://github.com/conda-forge/miniforge):

```bash
mamba env create -f environment.yml
mamba activate ml-template
pip install -e .
```

Pip-only alternative (the same path Docker and CI use — pulls CPU-only torch wheels):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Both install the `mlpipe` console command.

## Quickstart

Fetch the demo datasets (Titanic → classification, California Housing → regression):

```bash
python scripts/download_data.py --dataset all
```

Then drive everything through `mlpipe` (all commands take `--config` plus repeatable `--set`
overrides):

```bash
# Automated EDA report (prints the HTML path)
mlpipe eda --config configs/experiment/smoke_titanic.yaml

# Train one model end-to-end: validate -> split -> CV -> fit -> evaluate -> save bundle
mlpipe train --config configs/experiment/smoke_titanic.yaml

# Same run, different model — swap the whole model group from the CLI
mlpipe train --config configs/experiment/smoke_titanic.yaml --set model=lightgbm

# Optuna hyperparameter search, then re-fit + persist the best model
mlpipe tune --config configs/experiment/smoke_titanic.yaml --set tuning.n_trials=5

# Train every model in compare.models (+ voting/stacking) and print a leaderboard
mlpipe compare --config configs/experiment/smoke_titanic.yaml

# Re-evaluate a saved bundle on the held-out test split
mlpipe evaluate --config configs/experiment/smoke_titanic.yaml --bundle latest

# Serve the newest bundle over HTTP (FastAPI + uvicorn)
mlpipe serve --bundle latest --port 8000
```

Query the API with raw, Titanic-shaped records — preprocessing happens inside the bundle:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
       "Parch": 0, "Fare": 7.25, "Cabin": null, "Embarked": "S"}'

curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"records": [
        {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1,
         "Parch": 0, "Fare": 71.28, "Cabin": "C85", "Embarked": "C"},
        {"Pclass": 3, "Sex": "male", "Age": null, "SibSp": 0,
         "Parch": 0, "Fare": 8.05, "Cabin": null, "Embarked": "S"}
      ]}'
```

Browse tracked runs in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

The regression demo works identically: `configs/experiment/smoke_california.yaml`.

## Project structure

```
.
├── configs/                  # All behavior lives here (see Configuration)
│   ├── experiment/           # Entry points: compose the group configs below
│   ├── data/                 # Dataset definitions incl. full column schemas
│   └── model/  preprocessing/  features/  training/  tuning/  evaluation/  serving/
├── data/
│   ├── raw/                  # Input CSVs (gitignored; see scripts/download_data.py)
│   └── processed/
├── src/ml_pipeline/
│   ├── cli.py                # `mlpipe` entrypoint: eda|train|tune|compare|evaluate|serve
│   ├── config/               # YAML loader (group resolution, deep merge) + pydantic schema
│   ├── core/                 # Registry, PipelineBundle artifact, shared types
│   ├── data/                 # Loaders (csv/parquet), splitters, pandera validation
│   ├── eda/                  # Analyzer registry + HTML report builder
│   ├── preprocessing/        # Imputer/scaler/encoder registries -> ColumnTransformer
│   ├── features/             # Polynomial, pruning, selection, PCA pipeline
│   ├── models/               # classical/ deep/ ensembles/ behind MODEL_REGISTRY
│   ├── training/             # Trainer orchestration, CV, imbalance samplers
│   ├── tuning/               # Optuna study + best-model re-fit
│   ├── evaluation/           # Metrics, plots, leaderboard, SHAP/LIME, error analysis
│   ├── serving/              # FastAPI app built from bundle metadata
│   ├── tracking/             # MLflow helpers
│   └── utils/                # io, logging, seeding
├── artifacts/                # Run outputs (gitignored): runs/, eda/, leaderboards
├── scripts/download_data.py  # Demo dataset fetcher
├── tests/                    # pytest suite
├── environment.yml           # conda env (primary)
└── requirements.txt          # pip mirror (Docker/CI)
```

## Configuration

An experiment file (`configs/experiment/*.yaml`) is the single entry point. It references config
**groups** by name; each group name resolves to a file in the matching directory:

| Group key       | Directory                | Selects                                        |
| --------------- | ------------------------ | ---------------------------------------------- |
| `data`          | `configs/data/`          | Source, target, task, column schema, split     |
| `model`         | `configs/model/`         | Model registry key + params + imbalance policy |
| `preprocessing` | `configs/preprocessing/` | Imputers, scalers, encoders, per-column rules  |
| `features`      | `configs/features/`      | Polynomial, pruning, selection, PCA            |
| `training`      | `configs/training/`      | CV strategy and folds                          |
| `tuning`        | `configs/tuning/`        | Optuna trials, metric, search-space overrides  |
| `evaluation`    | `configs/evaluation/`    | Metrics, explainability, error analysis        |
| `serving`       | `configs/serving/`       | Bundle path, host, port                        |

A group value can be a name or an inline mapping; with a `_base_` key the mapping deep-merges over
the named group file:

```yaml
# configs/experiment/smoke_titanic.yaml (excerpt)
data: titanic                 # -> configs/data/titanic.yaml
model: random_forest
preprocessing:
  _base_: default             # start from configs/preprocessing/default.yaml ...
  column_overrides:
    Cabin: { encoder: frequency }   # ... and override just this
```

Precedence, lowest to highest: **group file < inline experiment mapping < `--set` overrides**.
`--set` takes dot paths with YAML-typed values (`--set tuning.n_trials=5`,
`--set model.params.n_estimators=300`); a bare group assignment (`--set model=lightgbm`) swaps the
entire group file. The merged result is validated by a strict pydantic schema — unknown keys are
an error, and cross-field validators reject nonsensical combinations up front.

## Using your own dataset

No code changes required — four steps:

1. **Drop your file** into `data/raw/` (CSV or Parquet).
2. **Describe it** in `configs/data/<name>.yaml` — source, target, task, and the full column
   schema. This one file drives pandera validation, preprocessing column grouping, and the
   serving request model:

   ```yaml
   source: { type: csv, path: data/raw/churn.csv }
   target: Churned
   task: classification
   split: { strategy: stratified, test_size: 0.15, val_size: 0.15 }
   columns:
     - { name: CustomerId, dtype: int, role: id }
     - { name: Churned, dtype: int, allowed_values: [0, 1] }
     - { name: Plan, dtype: category, allowed_values: [free, pro, business] }
     - { name: MonthlySpend, dtype: float, ge: 0 }
     - { name: SupportTickets, dtype: int, nullable: true, ge: 0 }
   ```

   Roles: `feature` (default), `id` / `drop` (excluded from modeling), `time` (for temporal
   splits).
3. **Compose an experiment** in `configs/experiment/<name>.yaml`:

   ```yaml
   run: { experiment_name: churn, seed: 42 }
   data: churn
   model: xgboost
   preprocessing: default
   features: default
   training: default
   tuning: default
   evaluation: default
   serving: default
   ```
4. **Run it**: `mlpipe eda --config configs/experiment/churn.yaml`, then `train` / `tune` /
   `compare` / `serve` exactly as in the Quickstart.

## Extending the pipeline

Every extension point follows the same pattern: one new file, one registry decorator, one import
in the subpackage `__init__` so registration runs. Duplicate keys raise immediately.

**Add a model** — subclass `SklearnModel` (only `_build_estimator` is required; fit/predict
plumbing is inherited), register it, and import the module in
`src/ml_pipeline/models/classical/__init__.py`:

```python
# src/ml_pipeline/models/classical/extra_trees.py
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from ml_pipeline.core.types import TaskType
from ml_pipeline.models.base import MODEL_REGISTRY
from ml_pipeline.models.sklearn_wrapper import SklearnModel


@MODEL_REGISTRY.register("extra_trees")
class ExtraTreesModel(SklearnModel):
    name = "extra_trees"
    supported_tasks = frozenset({TaskType.CLASSIFICATION, TaskType.REGRESSION})

    def _build_estimator(self):
        cls = ExtraTreesClassifier if self.task is TaskType.CLASSIFICATION else ExtraTreesRegressor
        return cls(random_state=self.seed, **self.params)
```

Then `--set model=extra_trees` just works (optionally add `configs/model/extra_trees.yaml` and a
`get_default_search_space` classmethod for tuning). Deep-learning skeletons for CNN/LSTM/
Transformer live in `src/ml_pipeline/models/deep/stubs.py` with step-by-step instructions.

**Add a preprocessing component** — register a factory in
`src/ml_pipeline/preprocessing/components.py` and reference it by key in YAML:

```python
@SCALER_REGISTRY.register("quantile")
def _quantile_scaler(**options: Any) -> QuantileTransformer:
    return QuantileTransformer(**options)
```

**Add a data source** — subclass `BaseLoader` in `src/ml_pipeline/data/loaders.py`, register with
`@LOADER_REGISTRY.register("sqlite")`, then use `source: { type: sqlite, path: ... }`.

**Add an EDA analyzer** — subclass `EdaAnalyzer` under `src/ml_pipeline/eda/analyzers/`, register
it in `EDA_REGISTRY`, import it in the analyzers `__init__`, and list its key under
`eda.analyzers`.

## Artifacts

Every `train` / `tune` / `compare` run writes one directory:

```
artifacts/runs/<run_id>/          # <run_id> = <timestamp>_<experiment>_<model>
├── bundle.joblib                 # THE serving artifact: preprocessor + features + model + metadata
├── metadata.json                 # human-readable copy of the bundle metadata
├── config_resolved.yaml          # fully merged config that produced this run
├── metrics.json                  # CV + validation + test metrics
├── error_analysis.csv            # worst predictions on the test split
└── plots/                        # confusion matrix / ROC / PR / importances / SHAP
```

`bundle.joblib` is the **single deployable unit** — its `predict()` accepts a raw DataFrame with
the original feature columns, so preprocessing and model versions can never diverge. Anywhere a
bundle is expected (`mlpipe evaluate --bundle`, `mlpipe serve --bundle`, `serving.bundle_path`),
you may pass a run directory, a direct `bundle.joblib` path, or `latest`, which resolves to the
newest bundle under `artifacts/runs/`.

## Docker

Serving is containerized; the image installs via `requirements.txt` — the same pip path CI
exercises:

```bash
docker compose up --build     # builds the image and serves the newest bundle on :8000
```

Bundles are mounted **read-only** into the container (`./artifacts` → `/app/artifacts:ro`); train
on the host, restart the container to pick up a new `latest`. Note: Docker is not installed on the
dev machine, so the Docker files are verified in CI via the identical `requirements.txt` install
path rather than local builds.

## Design notes

- **No ydata-profiling** — deliberately excluded due to its heavy, frequently-conflicting
  dependency pins; the built-in analyzer-based EDA report covers the same ground and stays inside
  the pinned dependency set.
- **Imbalance handling is fold-safe** — SMOTE & co. resample the transformed *training fold only*;
  samplers are never serialized into the bundle. Class weights are forwarded only to models that
  support them.
- **Time-series guardrails** — config validators enforce sane combinations at load time:
  `split.strategy: time` requires a `time_column`, time-series CV, and `shuffle: false`, and
  forbids resampling (which would break temporal order). Stratified splits/CV and imbalance
  strategies are rejected for regression tasks.

## Development

```bash
pytest                # run the test suite
ruff check src tests  # lint (line length 100, sorted imports)
ruff format src tests
```

License: MIT.
