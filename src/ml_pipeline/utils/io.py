"""Small IO helpers for YAML/JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict, raising a clear error when missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a dict to YAML (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    """Write a dict to JSON (parents created); non-serializable values become strings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
