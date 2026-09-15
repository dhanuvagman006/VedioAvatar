from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKERS_DIR = REPO_ROOT / "workers"
WEB_DIR = REPO_ROOT / "web"


def resolve(path: str | os.PathLike, base: Path = REPO_ROOT) -> Path:
    """Resolve a possibly relative path against ``base`` (the repo root by default)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def env_python(envs_dir: Path, name: str) -> Path:
    """Path of the interpreter inside a per-model virtual environment."""
    env = envs_dir / name
    if os.name == "nt":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"
