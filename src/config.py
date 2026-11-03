"""Load the project configuration from config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Return config.yaml as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
