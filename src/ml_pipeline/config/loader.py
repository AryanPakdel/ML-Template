"""YAML config loading: group resolution, deep merge, CLI overrides, validation.

An experiment file references config *groups* by name::

    # configs/experiment/titanic.yaml
    data: titanic              # -> configs/data/titanic.yaml
    model: xgboost             # -> configs/model/xgboost.yaml
    preprocessing: default

A group value may also be an inline mapping; with a ``_base_`` key it deep-merges
over the named group file. Precedence (lowest to highest):

    group file  <  inline experiment mapping  <  ``--set`` CLI overrides

Overrides use dot paths (``--set tuning.n_trials=5``); values are parsed with
``yaml.safe_load`` so ``true``/``5``/``[a,b]`` become real types. A single-segment
override naming a group (``--set model=lightgbm``) swaps the whole group file.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from ml_pipeline.config.schema import PipelineConfig
from ml_pipeline.utils.io import read_yaml, write_yaml

logger = logging.getLogger(__name__)

GROUP_KEYS = (
    "data",
    "model",
    "preprocessing",
    "features",
    "training",
    "tuning",
    "evaluation",
    "serving",
)

BASE_KEY = "_base_"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (lists are replaced, not merged)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_by_dotpath(data: dict[str, Any], dotpath: str, value: Any) -> None:
    """Set ``data['a']['b']['c'] = value`` for dotpath ``'a.b.c'``, creating dicts."""
    keys = dotpath.split(".")
    node = data
    for key in keys[:-1]:
        existing = node.get(key)
        if not isinstance(existing, dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``['a.b=1', 'model=lightgbm']`` into ``{dotpath: typed_value}``."""
    parsed: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like 'dot.path=value'")
        dotpath, raw_value = pair.split("=", 1)
        parsed[dotpath.strip()] = yaml.safe_load(raw_value)
    return parsed


def _resolve_group(
    group: str, value: Any, configs_root: Path
) -> Any:
    """Resolve one group entry to its final mapping."""
    if isinstance(value, str):
        return read_yaml(configs_root / group / f"{value}.yaml")
    if isinstance(value, dict) and BASE_KEY in value:
        inline = dict(value)
        base_name = inline.pop(BASE_KEY)
        base = read_yaml(configs_root / group / f"{base_name}.yaml")
        return deep_merge(base, inline)
    return value


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    configs_root: str | Path | None = None,
) -> PipelineConfig:
    """Load an experiment YAML into a validated :class:`PipelineConfig`.

    Args:
        path: experiment YAML (typically ``configs/experiment/<name>.yaml``).
        overrides: ``--set``-style strings, e.g. ``["tuning.n_trials=5"]``.
        configs_root: directory holding the group subdirectories; inferred as
            the experiment file's grandparent when omitted.
    """
    path = Path(path)
    raw = read_yaml(path)
    root = Path(configs_root) if configs_root is not None else path.parent.parent

    parsed = parse_overrides(overrides or [])

    # Whole-group swaps (e.g. model=lightgbm) apply before group resolution.
    for dotpath in list(parsed):
        if dotpath in GROUP_KEYS and isinstance(parsed[dotpath], str):
            raw[dotpath] = parsed.pop(dotpath)

    for group in GROUP_KEYS:
        if group in raw:
            raw[group] = _resolve_group(group, raw[group], root)

    for dotpath, value in parsed.items():
        _set_by_dotpath(raw, dotpath, value)

    config = PipelineConfig.model_validate(raw)
    logger.debug("Loaded config from %s (overrides=%s)", path, overrides)
    return config


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Write the fully-resolved config as YAML (for the run directory)."""
    write_yaml(config.model_dump(mode="json"), Path(path))
