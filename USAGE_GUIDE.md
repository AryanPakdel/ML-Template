# Roadmap: Using the ML Pipeline Template

A stage-by-stage guide from first run to shipping your own dataset. Commands assume
you are in the repo root. See `README.md` for full reference documentation.

## Stage 0 — One-time setup (5 min)

```bash
mamba activate ml-template          # env already exists; else: mamba env create -f environment.yml
pip install -e .                    # only needed on fresh clones
python scripts/download_data.py --dataset all
```

## Stage 1 — Learn the loop on the demos (30 min)

Run each command and open the artifact it prints:

1. `mlpipe eda --config configs/experiment/smoke_titanic.yaml` → open `report.html`
2. `mlpipe train --config configs/experiment/smoke_titanic.yaml` → inspect
   `artifacts/runs/<id>/` (metrics.json, plots/, bundle.joblib)
3. `mlpipe train --config configs/experiment/smoke_titanic.yaml --set model=lightgbm`
   → learn `--set` overrides (swap any config group from the CLI)
4. `mlpipe compare --config configs/experiment/smoke_titanic.yaml` → leaderboard + ensembles
5. `mlpipe tune --config configs/experiment/smoke_titanic.yaml --set tuning.n_trials=5`
6. `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` → browse every run
7. `mlpipe serve --bundle latest --port 8000` → curl `/predict` (payload examples in README)

Repeat once with `smoke_california.yaml` to see the regression path.

## Stage 2 — Understand the config system (1 hour)

Read in this order:

1. `configs/experiment/titanic.yaml` — the entry point
2. `configs/data/titanic.yaml` — the column schema (the single most important file)
3. One file each from `configs/model/`, `preprocessing/`, `training/`, `tuning/`

Rules: experiment files reference groups by name; an inline mapping with `_base_:`
deep-merges over the named group file; `--set dot.path=value` wins over everything.

## Stage 3 — Swap in YOUR dataset (the whole point)

1. Drop `data/raw/mydata.csv`
2. Write `configs/data/mydata.yaml`: source path, `target`, `task`, and the full
   `columns:` list (name / dtype / nullable / ge / le / allowed_values / role —
   `drop` for junk columns, `id` for identifiers)
3. Write `configs/experiment/mydata.yaml` referencing `data: mydata` plus default groups
4. Run the Stage 1 loop: **eda → train → compare → tune → serve**

Validation errors will tell you exactly which schema declarations are wrong —
fix the YAML, not the code.

## Stage 4 — Iterate like an ML engineer

- EDA report flags leakage / high cardinality / imbalance → adjust `preprocessing:`
  (encoders, `column_overrides`) and `model.imbalance` in config
- `compare` to shortlist models → `tune` the winner with more trials →
  `evaluate --bundle <path>` on the held-out test split
- Track everything in the MLflow UI; each run dir is self-contained and
  reproducible via its `config_resolved.yaml`

## Stage 5 — Extend the code (when config isn't enough)

Each extension is one new file + one `__init__` import (README "How to extend"
has code snippets):

- **New model** → subclass `SklearnModel`, decorate with `@MODEL_REGISTRY.register("name")`
- **New preprocessing step** → register a factory in `preprocessing/components.py`
- **New data source** → subclass `BaseLoader` in `data/loaders.py`
- **New EDA section** → subclass `EdaAnalyzer`
- **Vision/NLP later** → fill in `models/deep/stubs.py` following the MLP pattern

## Stage 6 — Ship

- `docker compose up` serves the latest bundle (mounts `artifacts/` read-only)
- Push to GitHub — CI runs ruff + pytest automatically
- Run `pytest -q` before any commit that touches `src/`
