"""Configuration loader: YAML -> frozen namespace with path resolution."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _resolve_paths(cfg: dict, project_root: Path) -> dict:
    """Convert relative path strings to absolute paths (resolved against project_root).

    Only converts keys that end with '_path', '_dir', '_root', '_cache', '_file',
    or that are named 'root' / 'label_mapping'.
    """
    path_keys = {
        "root", "label_mapping", "ddr_root", "odir_root", "preprocessed_cache",
        "pretrained_path", "log_dir", "output_dir",
    }

    def _walk(d: Any) -> Any:
        if isinstance(d, dict):
            result = {}
            for k, v in d.items():
                if k in path_keys and isinstance(v, str) and v:
                    # Don't convert absolute paths or empty strings
                    p = Path(v)
                    if not p.is_absolute() and v not in ("no", "false", ""):
                        v = str(project_root / v)
                result[k] = _walk(v)
            return result
        if isinstance(d, list):
            return [_walk(x) for x in d]
        return d

    return _walk(cfg)


def load_config(config_path: str | Path, cli_overrides: dict | None = None) -> SimpleNamespace:
    """Load a YAML config file, resolve paths, apply overrides, and return a frozen namespace.

    Args:
        config_path: Path to the YAML config file.
        cli_overrides: Optional dict of dotted-key overrides, e.g.
            {'training.stage1.epochs': 5, 'model.img_size': 256}.

    Returns:
        SimpleNamespace with all config values (read-only by convention).
    """
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent  # config.yaml -> configs/ -> project root

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _resolve_paths(cfg, project_root)

    # Apply CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            _set_nested(cfg, key, value)

    return _dict_to_namespace(cfg)


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(x) for x in d]
    return d


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set a nested dict value using a dotted key, converting value to appropriate type."""
    keys = dotted_key.split(".")
    d = cfg
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Try to infer type from existing value
    existing = d.get(keys[-1])
    if existing is not None and not isinstance(existing, type(value)):
        try:
            value = type(existing)(value)
        except (ValueError, TypeError):
            pass
    d[keys[-1]] = value


def parse_args() -> tuple[Path, dict]:
    """Parse command-line arguments for training scripts.

    Returns:
        (config_path, cli_overrides_dict)
    """
    parser = argparse.ArgumentParser(description="RetLesionUni Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Config overrides in key=value format (e.g., training.stage1.epochs=10).",
    )
    args = parser.parse_args()

    overrides = {}
    for item in args.overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            overrides[key] = _parse_override_value(value)

    return Path(args.config), overrides


def _parse_override_value(value: str) -> Any:
    """Parse a CLI override value to the appropriate Python type."""
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
