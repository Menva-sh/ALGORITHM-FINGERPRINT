"""Load the project configuration from config.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH, profile: str | None = None) -> dict[str, Any]:
    """Return config.yaml as a dict, optionally with a named profile applied."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_profile(cfg, profile) if profile else cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_profile(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Config with profiles[name] deep-merged over it (e.g. a separate large-graph dataset)."""
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(profiles)}")
    return _deep_merge(cfg, profiles[name])
