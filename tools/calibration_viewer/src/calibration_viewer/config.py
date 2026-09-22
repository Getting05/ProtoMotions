from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    required = ("base_link", "keypoint_links")
    missing = [key for key in required if key not in cfg.get("robot", {})]
    if missing:
        raise ValueError(f"Config robot section is missing: {missing}")
    return cfg


def save_config(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)

