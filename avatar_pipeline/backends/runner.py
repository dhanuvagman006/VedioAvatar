"""Run a worker script inside one of the per-model virtual environments and stream its output."""
from __future__ import annotations

import collections
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Sequence

from .base import BackendError, LogFn, ProgressFn

PROGRESS_PREFIX = "@@PROGRESS "


def run_worker(python: Path, script: Path, args: Sequence[str | os.PathLike], *, cwd: Path,
               log: LogFn, progress: ProgressFn, timeout: int | None = None,
               env_extra: dict[str, str] | None = None) -> None:
    if not python.exists():
        raise BackendError(
            f"Interpreter not found: {python}\n"
            "The per-model environments have not been created yet. Run: python scripts/setup_envs.py"
        )
    if not script.exists():
        raise BackendError(f"Worker script not found: {script}")

    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    # Never leak the orchestrator's virtualenv into the worker interpreter.
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)

    cmd = [str(python), str(script), *map(str, args)]
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    tail: collections.deque[str] = collections.deque(maxlen=40)

    def _pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            # tqdm rewrites the same line with carriage returns; keep only the last state.
            if "\r" in line:
                line = line.split("\r")[-1]
            if not line.strip():
                continue
            if line.startswith(PROGRESS_PREFIX):
                try:
                    payload = json.loads(line[len(PROGRESS_PREFIX):])
                    progress(float(payload.get("fraction", 0.0)), str(payload.get("message", "")))
                except (ValueError, TypeError):
                    log(line)
                continue
            tail.append(line)
            log(line)

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(timeout=5)
        raise BackendError(f"Worker timed out after {timeout}s: {script.name}")
    reader.join(timeout=30)
    if proc.returncode != 0:
        raise BackendError(
            f"Worker {script.name} exited with code {proc.returncode}. Last output:\n" + "\n".join(tail)
        )
