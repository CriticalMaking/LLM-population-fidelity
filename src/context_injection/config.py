from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("context_injection_config.json")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    return {
        key: _resolve(base, value)
        if key.endswith(("_csv", "_root")) or key == "data_root"
        else value
        for key, value in data.items()
    }


def wvs_csv(path: Path = DEFAULT_CONFIG) -> Path:
    config = load_config(path)
    return Path(config["wvs_csv"])
