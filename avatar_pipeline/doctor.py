"""Environment self-check used by `avatar doctor` and /api/health."""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .backends import get_lipsync_backend, get_tts_backend
from .config import Config
from .gpu import gpu_info
from .media import MediaError, find_binaries
from .paths import env_python


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _probe_env(python: Path, timeout: int = 90) -> str:
    code = ("import torch, sys; print(f'python {sys.version.split()[0]}, torch {torch.__version__}, "
            "cuda={torch.cuda.is_available()}' + (', ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))")
    try:
        proc = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return "timed out importing torch"
    if proc.returncode != 0:
        return "broken: " + (proc.stderr.strip().splitlines() or ["?"])[-1]
    return proc.stdout.strip()


def quick_checks(cfg: Config) -> list[Check]:
    """Fast checks (no subprocesses) suitable for the health endpoint."""
    checks: list[Check] = []
    try:
        ff, fp = find_binaries(cfg.get("ffmpeg.binary") or None, cfg.get("ffmpeg.ffprobe") or None)
        checks.append(Check("ffmpeg", True, f"{ff} | {fp}"))
    except MediaError as exc:
        checks.append(Check("ffmpeg", False, str(exc)))
    gpu = gpu_info()
    checks.append(Check("gpu", gpu is not None,
                        f"{gpu['name']}, {gpu['vram_total_mb']} MB VRAM" if gpu else "nvidia-smi not found"))
    for kind, name in (("tts", "chatterbox"), ("lipsync", "wav2lip"), ("lipsync", "latentsync")):
        backend = get_tts_backend(name, cfg) if kind == "tts" else get_lipsync_backend(name, cfg)
        problems = backend.check()
        checks.append(Check(f"{kind}:{name}", not problems, "ready" if not problems else "; ".join(problems)))
    return checks


def full_checks(cfg: Config) -> list[Check]:
    """Quick checks plus importing torch inside every per-model environment."""
    checks = [Check("python", True, f"{sys.version.split()[0]} on {platform.platform()}")]
    checks += quick_checks(cfg)
    for env in ("tts", "wav2lip", "latentsync"):
        py = env_python(cfg.envs_dir, env)
        if py.exists():
            detail = _probe_env(py)
            checks.append(Check(f"env:{env}", not detail.startswith(("broken", "timed out")), detail))
        else:
            checks.append(Check(f"env:{env}", False, f"not created ({py})"))
    return checks


def format_checks(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks) + 2
    return "\n".join(f"{'OK  ' if c.ok else 'FAIL'} {c.name.ljust(width)} {c.detail}" for c in checks)
