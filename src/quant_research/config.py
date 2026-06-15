from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/strategy.yaml") -> dict[str, Any]:
    """Load a YAML strategy configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: " + str(config_path))

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def ensure_output_dirs(config: dict[str, Any]) -> Path:
    """Create standard output folders and return the base output path."""
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    for subdir in ["plots", "reports", "data"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir
