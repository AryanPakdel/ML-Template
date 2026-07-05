# Learning Path: How Every File Works

A file-by-file walkthrough of the codebase for studying it, ordered so each layer only
depends on layers you've already read. Work through the 8 layers in order; each ends with
the concept you should take away and a hands-on exercise.

**The big picture first.** One validated config object drives everything. Every swappable
piece (loader, splitter, imputer, encoder, model, sampler, CV strategy, EDA analyzer) is a
class or factory registered under a string key; config names the key, orchestration looks
it up. Training produces exactly one artifact — `bundle.joblib` — and serving loads exactly
that. The full data flow:

```
CSV ──load──▶ raw df ──pandera──▶ validated df ──split──▶ train/val/test
      train ──[fit: preprocessor ▶ features ▶ (sampler) ▶ model]──▶ fitted objects
      val/test ──[transform only]──▶ metrics, plots, SHAP, error analysis
      fitted objects + metadata ──▶ PipelineBundle.save() ──▶ bundle.joblib
      serving: raw JSON ──▶ pydantic 422 ──▶ range-check 400 ──▶ bundle.predict()
```

---

## Layer 1 — The kernel: `src/ml_pipeline/core/`

### `core/registry.py`
One generic class, `Registry(Generic[T])`, holding a `dict[str, T]`:
- `register(key)` — returns a decorator; raises on duplicate keys.
- `get(key)` — lookup; the `KeyError` message lists all available keys (so a config typo
  is self-explaining).
- `available()`, `__contains__`, `__len__`.

**Takeaway:** the registry/factory pattern. Ten registries across the codebase are all
instances of this one class. This is what makes "add a model = one file + one import".

### `core/types.py`
Shared vocabulary, no logic:
- `TaskType` (classification/regression), `ExplainerHint` (tree/linear/kernel/none — which
  SHAP explainer suits a model), `ColumnRole` (feature/id/drop) — all `StrEnum`s so they
  serialize as plain strings.
- `DatasetSplits` — frozen dataclass of three DataFrames (train/val/test) with `sizes()`.

**Takeaway:** stages never import each other; they share only these types + config objects.

### `core/artifacts.py`
The training→inference seam:
- `BundleMetadata` (dataclass) — run_id, task, target, `feature_columns` (ordered raw input
  names), `raw_feature_schema` (ColumnSpec dicts — feeds the serving request model),
  `class_labels` (index→original label), metrics, full config.
- `PipelineBundle` — holds fitted `preprocessor`, `feature_pipeline`, `model`, metadata.
  - `_select_features(df)` — subset/reorder raw input, loud error on missing columns.
  - `transform(df)` — preprocessor → feature pipeline, exactly as at fit time.
  - `predict(df)` — transform → model.predict → decode class indices back to original
    labels via `class_labels`.
  - `save(run_dir)` / `load(path)` — one `joblib.dump`/`load`.
- `new_run_id()` (timestamped), `create_run_dir()` (collision-suffixing),
  `resolve_bundle_path("latest" | dir | file)` — "latest" = newest bundle by mtime.

**Takeaway:** zero train/serve skew comes from persisting the *fitted objects*, never
re-declaring preprocessing at inference. **Exercise:** load a bundle in a REPL and call
`bundle.transform()` on 3 raw rows; inspect the output columns.

---

## Layer 2 — Config: `src/ml_pipeline/config/`

### `config/schema.py`
The entire pydantic v2 hierarchy; every model inherits `StrictModel`
(`extra="forbid"` → unknown YAML keys fail at load, not silently ignore).
Read top-to-bottom: `DataSourceConfig`, `ColumnSpec` (name/dtype/nullable/ge/le/
allowed_values/role — the one schema that feeds pandera, preprocessing grouping, AND the
serving API), `SplitConfig`, `DataConfig` (with `feature_columns()` /
`columns_by_role()` helpers), `EdaConfig`, `PreprocessingConfig` (numeric/categorical/
datetime + `column_overrides`), `FeatureConfig`, `ImbalanceConfig`, `ModelConfig`,
`CvConfig`, `TuningConfig` + `SearchSpaceParam`, `EvaluationConfig`, `CompareConfig`,
`RunConfig`/`MlflowConfig`, `ServingConfig`, and root `PipelineConfig`.
The root's `_cross_field_rules` validator encodes ML sanity: time split ⇒ timeseries CV,
no shuffle, no SMOTE; stratified/imbalance ⇒ classification only.

**Takeaway:** validated config as the single source of truth; cross-field validators
encode domain rules so invalid experiments can't even start.

### `config/loader.py`
- `load_config(path, overrides)` — 4 steps: read experiment YAML → apply whole-group `--set`
  swaps (`model=lightgbm`) → resolve each group key (string → `configs/<group>/<name>.yaml`;
  mapping with `_base_:` → deep-merge over the group file) → apply dot-path overrides →
  `PipelineConfig.model_validate`.
- `deep_merge` (recursive, lists replaced), `_set_by_dotpath`, `parse_overrides` (values
  typed via `yaml.safe_load` so `5`/`true`/`[a,b]` become real types), `dump_config`.

**Takeaway:** config composition (what Hydra does) in ~150 readable lines.
**Exercise:** add `--set training.cv.n_splits=3` and diff the printed `config_resolved.yaml`.

---

## Layer 3 — Utils: `src/ml_pipeline/utils/`
- `seed.py` — `set_global_seed()`: PYTHONHASHSEED, `random`, numpy, torch (lazy import).
- `logging.py` — `setup_logging(level, log_file)`: idempotent root-logger config, console +
  file handlers, quiets noisy third-party loggers.
- `io.py` — `ensure_dir`, `read_yaml`/`write_yaml`, `read_json`/`write_json`.

---

## Layer 4 — Data: `src/ml_pipeline/data/` + `scripts/download_data.py`

### `data/loaders.py`
`BaseLoader` ABC (`load(cfg) -> DataFrame`), `CsvLoader`/`ParquetLoader` registered in
`LOADER_REGISTRY`, and `load_dataframe(cfg)` dispatching by `source.type`.

### `data/validation.py`
- Guarded `import pandera.pandas as pa` (pandera ≥0.24 moved the pandas backend).
- `build_pandera_schema(data_cfg)` — compiles each `ColumnSpec` into a `pa.Column`
  (dtype coercion, ge/le/isin checks, nullability).
- `validate_dataframe(df, data_cfg)` — validates with `lazy=True` (collects ALL failures),
  re-raises as `DataValidationError` with a per-column human-readable summary.

**Takeaway:** declarative data contracts; fail loudly and completely, not on the first error.

### `data/splitters.py`
`BaseSplitter` ABC + three registered strategies: `random` (two-stage `train_test_split`
with val fraction rescaled by `val/(1-test)` so final proportions match), `stratified`
(same, stratifying on the target at both stages), `time` (sort by `time_column`, slice
sequentially — no shuffle ever). `split_dataset()` dispatches.

**Exercise:** prove to yourself why the val fraction must be rescaled after the test split.

---

## Layer 5 — Transform stages

### `preprocessing/transformers.py` — custom sklearn transformers
All implement `fit`/`transform`/`get_feature_names_out` so they compose in Pipelines:
- `DatetimeFeatureExtractor` — `to_datetime(errors="coerce")`, emits `<col>_year/month/...`,
  drops originals.
- `FrequencyEncoder` — fits value→relative-frequency map; unseen/NaN → 0.0. (Leakage-safe
  because our CV loop refits it per fold.)
- `OutlierClipper` — learns IQR bounds at fit, clips at transform.

### `preprocessing/components.py`
Three registries of **factory functions** `(options) -> transformer`:
imputers (mean/median/most_frequent/constant/knn), scalers (standard/minmax/robust/none),
encoders (onehot — dense, `handle_unknown="ignore"`; ordinal; **target** — sklearn's
internally cross-fitted `TargetEncoder`, leakage-safe by construction; frequency).

### `preprocessing/builder.py`
`build_preprocessor(prep_cfg, data_cfg, train_df=None)`:
1. Groups feature columns by **declared** dtype (int/float→numeric, category/string/bool→
   categorical, datetime→datetime).
2. Numeric pipe: imputer → optional clipper → scaler. Categorical pipe: imputer → encoder.
   Datetime pipe: extractor → imputer → scaler.
3. Cardinality guard: onehot column with `nunique > max_onehot_cardinality` in `train_df`
   degrades to frequency encoding (warning).
4. `column_overrides` get their own ColumnTransformer entries; identical choices share one.
5. Returns `Pipeline([("preprocess", ColumnTransformer(...))]).set_output(transform="pandas")`
   — pandas output is why feature names survive into SHAP/importance plots.
Row-dropping ops (`outliers.method="remove"`) deliberately live in the trainer, not here —
a persisted transformer must never change row counts at inference.

### `features/transformers.py` + `features/builder.py`
`CorrelationPruner` (drops the later member of each |corr|>threshold pair),
`ImportanceSelector` (quick RandomForest ranks features; keep top-k or ≥median),
`LoggingPCA` (PCA that logs cumulative explained variance).
`build_feature_pipeline(cfg, task, seed)` chains polynomial → pruning → selection → PCA,
returning `None` when everything is disabled.

**Takeaway:** everything post-ingestion is "just" sklearn transformers — the composability
comes from respecting the `fit/transform/get_feature_names_out` protocol.

---

## Layer 6 — Models: `src/ml_pipeline/models/`

### `models/base.py` — THE contract (read this file completely)
`BaseModel` ABC: ClassVars `name`, `supported_tasks`, `explainer_hint`;
`__init__(params, task, seed)`; abstract `fit(X, y, X_val, y_val, sample_weight)` and
`predict`; default `predict_proba` raises `NotImplementedError`; `get_feature_importance`;
classmethod `get_default_search_space(trial, task)` for Optuna. Models receive
**post-preprocessing matrices only** — that's what keeps them dataset-agnostic.
`MODEL_REGISTRY` lives here.

### `models/sklearn_wrapper.py`
`SklearnModel(BaseModel)` — the adapter that makes classical models ~15 lines each:
subclasses implement only `_build_estimator()`; the wrapper handles fitting (forwarding
`sample_weight` only if the estimator's `fit` signature accepts it — see `inspect.signature`),
`predict/predict_proba` delegation, and importance extraction (`feature_importances_` or
`|coef_|`). Hook `_fit_kwargs(X_val, y_val)` lets boosting models add `eval_set`.

### `models/classical/*.py` (9 files)
Each: `@MODEL_REGISTRY.register("<key>")`, task support, explainer hint, estimator
construction switching on `self.task`, and an Optuna search space. Notables:
`linear.py` holds both logistic (clf-only, LINEAR) and linear regression (reg-only);
xgboost/lightgbm/catboost pass `eval_set` for early stopping and are silenced
(`verbosity=0` / `verbose=-1` / `verbose=0, allow_writing_files=False`); `svm.py` forces
`probability=True` on SVC; heavy imports are lazy inside `_build_estimator`.

### `models/deep/` — the DL track behind the SAME contract
- `tabular_data.py` — `TabularDataset` (float32 tensors; long targets for clf) +
  `build_dataloaders()`.
- `mlp.py` — `MLPModel(BaseModel)`. Inside: `_TabularMLP(LightningModule)` (Linear/ReLU/
  Dropout stack, CE or MSE loss, AdamW + ReduceLROnPlateau on val_loss). `fit()` converts
  numpy→loaders, carves a 10% val split if none given, runs `lightning.Trainer` with
  EarlyStopping + ModelCheckpoint + optional 16-mixed precision + gradient clipping, then
  reloads the best checkpoint. **Study `__getstate__`/`__setstate__`:** weights are
  serialized to bytes via `torch.save` into a BytesIO so the whole model survives
  `joblib.dump` inside the bundle (the Lightning Trainer and any tuner callbacks are
  dropped from the pickle).
- `stubs.py` — unregistered CNN/LSTM/Transformer skeletons documenting the plug-in recipe.

### `models/ensembles/`
- `voting.py` — trains each base (looked up in the registry, so MLP can be a base),
  soft-votes averaged probabilities with weights; falls back to weighted hard voting.
- `stacking.py` — builds **out-of-fold** base predictions with StratifiedKFold/KFold
  (never lets the meta-learner see in-fold predictions — that would be leakage), fits a
  LogisticRegression/RidgeCV meta-learner on them, then refits bases on all data.

**Takeaway:** one interface makes XGBoost, a Lightning network, and a stack of both
interchangeable everywhere downstream. **Exercise:** register a `Ridge` model in a new
file and run it through `mlpipe compare`.

---

## Layer 7 — Orchestration

### `training/imbalance.py`
`SAMPLER_REGISTRY` (smote/random_over/random_under factories), `apply_sampler()`
(train-fold-only; falls back gracefully when SMOTE can't run on tiny folds),
`balanced_sample_weight()` for the `class_weight` strategy.

### `training/cross_validation.py` — the leakage-killer (most important file to study)
- `CV_SPLITTER_REGISTRY` — kfold/stratified/timeseries.
- `prepare_features_and_target(df, cfg, class_labels=None)` — selects feature columns,
  label-encodes y against a *fixed* label set (pass labels from the full dataset so every
  split shares one mapping — subtle bug this prevents: a split missing one class).
- `filter_training_rows()` — IQR outlier-row removal, training data only.
- `fit_fold(cfg, X_tr, y_tr, ...)` — THE shared fit path: row filter → build+fit
  preprocessor → build+fit feature pipeline → sampler → model. Used by both CV folds and
  the final fit, so the two can never drift.
- `cross_validate()` — per fold: refit everything on the fold's train part only, transform
  the val part, score. Aggregates mean±std.

**Takeaway:** the only reliable way to prevent preprocessing leakage is to refit the
whole transform stack inside every fold. sklearn's `cross_val_score` can do this for pure
sklearn pipelines; we hand-roll the loop so DL models and samplers join too.

### `training/trainer.py`
`load_validated_frame(cfg)` (load → validate → drop role=drop) and `PipelineTrainer.run()`:
seed → registry/task check → load/split → MLflow run → optional `cross_validate` on
train+val → final `fit_fold` on train (val for early stopping) → evaluate val+test →
plots → error analysis → SHAP (lazy import, best-effort) → build `BundleMetadata` →
`bundle.save()` → `metrics.json`/`config_resolved.yaml` → MLflow artifacts. Returns
`TrainResult`.

### `tuning/optuna_tuner.py` + `tuning/callbacks.py`
The objective merges the model's own `get_default_search_space(trial)` with YAML
`search_space` overrides, runs `cross_validate`, returns the primary metric; actual params
are stashed via `trial.set_user_attr` (needed because MLP's `hidden_dims` is composed from
several suggests). MedianPruner + TPESampler(seed). Each trial = nested MLflow run. Best
params are re-fit through the normal `PipelineTrainer`. `OptunaPruningCallback` is a
~20-line Lightning callback reporting `val_loss` per epoch with a monotonic step counter
(so pruning works across CV folds).

### `evaluation/` (metrics, plots, error_analysis, explain, leaderboard)
- `metrics.py` — `compute_metrics(task, y_true, y_pred, y_proba)` (weighted P/R/F1,
  ROC-AUC/PR-AUC guarded for single-class folds; MAE/RMSE/R²/MAPE),
  `resolve_primary_metric` ("auto" → f1/rmse), `METRIC_DIRECTIONS`, clustering helpers.
- `plots.py` — confusion/ROC/PR/residuals/pred-vs-actual/importance; every function is
  guarded (a plot failure warns, never crashes a run).
- `error_analysis.py` — `worst_rows()` ranks test rows by |residual| (reg) or misclassified
  confidence (clf).
- `explain.py` — dispatch on `explainer_hint`: TreeExplainer / LinearExplainer /
  KernelExplainer (sample-capped — kernel SHAP is O(background × eval × features));
  `_to_matrix` normalizes SHAP's three output shapes; any failure falls back to LIME HTML,
  then to skipping. Explainability never fails a run.
- `leaderboard.py` — `run_comparison()`: every `compare.models` entry (+ voting/stacking,
  which are just registered models) goes through the same `PipelineTrainer`; ranks by
  val primary metric (direction-aware), writes CSV + hand-rolled markdown, SHAP-explains
  the winner.

### `tracking/mlflow_utils.py`
Fluent-API-only helpers: `init_tracking` (SQLite URI), `start_run` contextmanager,
`log_config` (flattens nested dict to dotted params, truncated), `log_metrics`,
`log_artifacts_dir`. Every helper swallows+warns on MLflow errors — tracking must never
sink a training run.

---

## Layer 8 — The edges

### `serving/schemas.py` + `serving/app.py`
- `build_request_model(metadata)` — `pydantic.create_model` from the bundle's
  `raw_feature_schema`; category field types derive from `allowed_values` (Pclass=[1,2,3]
  → int); nullable → `type | None = None`. Free 422s from FastAPI.
- `check_value_constraints()` — manual ge/le/allowed checks → 400 with readable messages
  (deliberately not importing the data stage; serving depends only on the bundle).
- `create_app(bundle_spec, artifacts_dir)` — loads bundle once; `/health`, `/model_info`,
  `/predict`, `/predict_batch`; `_coerce_to_training_dtypes` re-aligns JSON types with
  what the fitted preprocessor saw; note the `__annotations__` injection trick that lets
  FastAPI resolve dynamically created request models under PEP 563.

### `eda/` — base.py (EdaAnalyzer ABC, `EdaSection`, `fig_to_html` base64 embedding,
EDA_REGISTRY), 8 analyzers in `eda/analyzers/` (missing, distributions, categoricals,
correlation + target ranking, class_balance, IQR/z-score outliers, cardinality, leakage
screening via near-perfect target correlation), and `report.py` (`run_eda` → self-contained
HTML + markdown findings summary).

### `cli.py`
argparse subcommands eda/train/tune/compare/evaluate/serve; heavy stages imported *inside*
command functions so `--help` is instant; every command = load_config → setup_logging →
delegate to one Layer-7 entry point.

### Root files
`environment.yml` (conda-forge, primary) / `requirements.txt` (exact-pinned pip mirror,
CPU-torch index — used by `Dockerfile` two-layer build and `.github/workflows/ci.yml`),
`pyproject.toml` (src-layout package, `mlpipe` entry point, ruff/pytest config),
`tests/` (46 tests — read `conftest.py` first: synthetic-data fixtures, everything
tmp_path-isolated, MLflow disabled).

---

## Suggested study schedule

| Session | Read | Then do |
|---|---|---|
| 1 | Layers 1–2 (registry, types, artifacts, schema, loader) | Trace `load_config` in a REPL |
| 2 | Layer 4 + `preprocessing/` | Break the Titanic schema on purpose; read the error |
| 3 | `models/base.py`, `sklearn_wrapper.py`, 2 classical files | Register your own Ridge model |
| 4 | `cross_validation.py` + `trainer.py` (the core) | Step through one `fit_fold` in a debugger |
| 5 | `models/deep/mlp.py` + `ensembles/` | joblib-dump an MLP, reload, predict |
| 6 | `tuning/` + `evaluation/` | Add a custom `search_space` and run tune |
| 7 | `serving/` + `cli.py` + tests | Add a `/predict_proba`-only endpoint; write its test |

## Call-stack trace to keep beside you (what `mlpipe train` actually runs)

```
cli.cmd_train
└─ PipelineTrainer.run()                        training/trainer.py
   ├─ set_global_seed                           utils/seed.py
   ├─ MODEL_REGISTRY.get(name)                  models/base.py ← registered by models/*/__init__.py imports
   ├─ load_validated_frame
   │  ├─ load_dataframe                         data/loaders.py   (LOADER_REGISTRY)
   │  └─ validate_dataframe                     data/validation.py (pandera, lazy)
   ├─ split_dataset                             data/splitters.py (SPLITTER_REGISTRY)
   ├─ cross_validate(train+val)                 training/cross_validation.py
   │  └─ per fold: fit_fold → transform val → compute_metrics
   ├─ fit_fold(train, val)                      ← same function as CV: no drift possible
   │  ├─ filter_training_rows                   (outlier "remove", train only)
   │  ├─ build_preprocessor().fit               preprocessing/builder.py
   │  ├─ build_feature_pipeline().fit           features/builder.py
   │  ├─ apply_sampler / balanced_sample_weight training/imbalance.py
   │  └─ Model(params, task, seed).fit
   ├─ compute_metrics (val, test)               evaluation/metrics.py
   ├─ plots / worst_rows / explain_model        evaluation/{plots,error_analysis,explain}.py
   ├─ PipelineBundle(...).save(run_dir)         core/artifacts.py
   └─ mlflow_utils.log_*                        tracking/mlflow_utils.py
```
