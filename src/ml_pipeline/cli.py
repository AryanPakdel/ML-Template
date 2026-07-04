"""``mlpipe`` command-line interface: eda | train | tune | compare | evaluate | serve.

Heavy stages are imported inside their command functions so ``mlpipe --help``
stays fast and each command only pays for what it uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from ml_pipeline.config.loader import load_config
from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _add_config_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--config", type=Path, required=required, help="Experiment YAML (configs/experiment/*.yaml)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOT.PATH=VALUE",
        help="Config override, repeatable (e.g. --set model=lightgbm --set tuning.n_trials=5)",
    )


def _load(args: argparse.Namespace, command: str) -> PipelineConfig:
    """Load config and initialize logging (console + timestamped file)."""
    cfg = load_config(args.config, overrides=args.overrides)
    log_file = Path(cfg.run.logs_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{command}.log"
    setup_logging(cfg.run.log_level, log_file)
    logger.info("Loaded config %s (overrides=%s)", args.config, args.overrides)
    return cfg


# ------------------------------------------------------------------ commands


def cmd_eda(args: argparse.Namespace) -> int:
    """Generate the automated EDA report."""
    cfg = _load(args, "eda")
    from ml_pipeline.eda.report import run_eda
    from ml_pipeline.training.trainer import load_validated_frame
    from ml_pipeline.utils.seed import set_global_seed

    set_global_seed(cfg.run.seed)
    df = load_validated_frame(cfg)
    out_dir = (
        Path(cfg.run.artifacts_dir)
        / "eda"
        / cfg.run.experiment_name
        / time.strftime("%Y%m%d_%H%M%S")
    )
    report_path = run_eda(df, cfg, out_dir)
    logger.info("EDA report: %s", report_path)
    print(report_path)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the configured model end-to-end and persist the bundle."""
    cfg = _load(args, "train")
    from ml_pipeline.training.trainer import PipelineTrainer

    result = PipelineTrainer(cfg).run()
    print(json.dumps({"run_id": result.run_id, "bundle": str(result.bundle_path),
                      "metrics": result.metrics}, indent=2, default=str))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Run the Optuna study, then re-fit and persist the best model."""
    cfg = _load(args, "tune")
    from ml_pipeline.tuning.optuna_tuner import OptunaTuner

    result = OptunaTuner(cfg).tune()
    print(json.dumps({
        "model": result.model_name,
        "metric": result.metric,
        "best_value": result.best_value,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "best_run_id": result.best_run.run_id,
        "bundle": str(result.best_run.bundle_path),
    }, indent=2, default=str))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Train all compare.models (+ ensembles) and print the leaderboard."""
    cfg = _load(args, "compare")
    from ml_pipeline.evaluation.leaderboard import run_comparison

    leaderboard, out_dir = run_comparison(cfg)
    print(leaderboard.to_string(index=False))
    print(f"\nLeaderboard artifacts: {out_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a saved bundle on the held-out test split of the configured data."""
    cfg = _load(args, "evaluate")
    from ml_pipeline.core.artifacts import PipelineBundle, resolve_bundle_path
    from ml_pipeline.data.splitters import split_dataset
    from ml_pipeline.evaluation.metrics import compute_metrics
    from ml_pipeline.training.cross_validation import (
        predict_proba_or_none,
        prepare_features_and_target,
    )
    from ml_pipeline.training.trainer import load_validated_frame

    bundle_spec = args.bundle or cfg.serving.bundle_path
    bundle = PipelineBundle.load(resolve_bundle_path(bundle_spec, cfg.run.artifacts_dir))
    logger.info("Evaluating bundle %s (model=%s)", bundle.metadata.run_id, bundle.metadata.model_name)

    df = load_validated_frame(cfg)
    splits = split_dataset(df, cfg.data, cfg.run.seed)
    X_test, y_test, _ = prepare_features_and_target(
        splits.test, cfg, bundle.metadata.class_labels
    )
    X_t = bundle.transform(X_test)
    y_pred = bundle.model.predict(X_t)
    y_proba = predict_proba_or_none(bundle.model, X_t, cfg.data.task)
    metrics = compute_metrics(cfg.data.task, y_test, y_pred, y_proba)
    print(json.dumps({"run_id": bundle.metadata.run_id, "test_metrics": metrics}, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve a trained bundle over HTTP (FastAPI + uvicorn)."""
    import uvicorn

    from ml_pipeline.serving.app import create_app

    host, port, bundle_spec, artifacts_dir = args.host, args.port, args.bundle, "artifacts"
    if args.config is not None:
        cfg = _load(args, "serve")
        host = host or cfg.serving.host
        port = port or cfg.serving.port
        bundle_spec = bundle_spec or cfg.serving.bundle_path
        artifacts_dir = cfg.run.artifacts_dir
    else:
        setup_logging("INFO")
    app = create_app(bundle_spec or "latest", artifacts_dir=artifacts_dir)
    uvicorn.run(app, host=host or "0.0.0.0", port=port or 8000, log_level="info")
    return 0


# ------------------------------------------------------------------ entrypoint


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argparse tree for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="mlpipe", description="Config-driven end-to-end ML pipeline template."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, doc in (
        ("eda", cmd_eda, "Generate the automated EDA report"),
        ("train", cmd_train, "Train one model end-to-end"),
        ("tune", cmd_tune, "Optuna hyperparameter search + best-model re-fit"),
        ("compare", cmd_compare, "Train all compare.models and build a leaderboard"),
        ("evaluate", cmd_evaluate, "Evaluate a saved bundle on the test split"),
    ):
        p = sub.add_parser(name, help=doc)
        _add_config_args(p, required=True)
        if name == "evaluate":
            p.add_argument("--bundle", default=None, help="Bundle path or 'latest'")
        p.set_defaults(fn=fn)

    serve = sub.add_parser("serve", help="Serve a trained bundle over HTTP")
    _add_config_args(serve, required=False)
    serve.add_argument("--bundle", default=None, help="Bundle path or 'latest'")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(fn=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (installed as ``mlpipe``)."""
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
