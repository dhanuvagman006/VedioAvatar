"""Helpers shared by the worker scripts. Must stay free of third-party imports."""
from __future__ import annotations

import json
import os
import subprocess
import sys

PROGRESS_PREFIX = "@@PROGRESS "


def log(message: str) -> None:
    print(message, flush=True)


def progress(fraction: float, message: str = "") -> None:
    payload = {"fraction": max(0.0, min(1.0, float(fraction))), "message": message}
    print(PROGRESS_PREFIX + json.dumps(payload), flush=True)


def fail(message: str, code: int = 2) -> None:
    print("ERROR: " + message, file=sys.stderr, flush=True)
    sys.exit(code)


def env_with_ffmpeg(ffmpeg_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if ffmpeg_dir:
        env["PATH"] = str(ffmpeg_dir) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run(cmd: list, cwd, env: dict[str, str]) -> int:
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd), env=env)
