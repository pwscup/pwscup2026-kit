"""採点・検証パラメータ（YAML）のローダ。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML を辞書として読み込む。"""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
