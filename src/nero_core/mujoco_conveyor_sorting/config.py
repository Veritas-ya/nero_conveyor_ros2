"""Configuration loading with explicit defaults and recursive overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(document).__name__}")
    return document


def load_project_config(
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    config: dict[str, Any] = {"config_dir": str(root.resolve())}
    for name in ("scene", "robot", "objects", "cameras", "latency", "runtime"):
        path = root / f"{name}.yaml"
        if path.is_file():
            config[name] = load_yaml(path)
        else:
            config[name] = {}
    if overrides:
        config = deep_merge(config, overrides)
    return config
