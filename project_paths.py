"""Portable paths for non-versioned data and model artifacts.

Code and documentation belong in Git. Large data and trained artifacts must be
provided through a shared location selected by environment variables, not a PC-
specific absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_DIR = _path_from_env("KEIBA_DATA_DIR", PROJECT_ROOT / "data")
MODEL_DIR = _path_from_env("KEIBA_MODEL_DIR", PROJECT_ROOT / "models")
DB_PATH = _path_from_env("KEIBA_DB_PATH", DATA_DIR / "keiba.db")
REGISTRY_DB_PATH = _path_from_env("KEIBA_REGISTRY_DB_PATH", DATA_DIR / "research_registry.db")


def model_path(filename: str) -> Path:
    return MODEL_DIR / filename
