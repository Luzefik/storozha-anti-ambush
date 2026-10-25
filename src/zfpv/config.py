"""Load config/default.yaml and apply dotted-key CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config/default.yaml")


def load_config(path: Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> dict[str, Any]:
    """Return the config as a nested dict.

    overrides: strings like "frames.ext=jpg" or "eval.max_detections=300".
    Values are parsed with YAML, so numbers, booleans, null and lists work.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key.subkey=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"Unknown config section {part!r} in override {item!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"Unknown config key {key!r} (typo?)")
        node[leaf] = yaml.safe_load(raw_value)
    return cfg
