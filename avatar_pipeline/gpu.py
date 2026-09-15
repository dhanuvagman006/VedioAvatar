from __future__ import annotations

import shutil
import subprocess
from typing import Any


def gpu_info() -> dict[str, Any] | None:
    """Query the first NVIDIA GPU via nvidia-smi. Returns None when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    try:
        total = int(float(parts[1]))
        used = int(float(parts[2]))
    except ValueError:
        return None
    return {
        "name": parts[0],
        "vram_total_mb": total,
        "vram_used_mb": used,
        "vram_free_mb": max(total - used, 0),
        "driver": parts[3] if len(parts) > 3 else "",
    }


def choose_lipsync_backend(requested: str, min_vram_for_latentsync_mb: int) -> str:
    """Resolve ``auto`` to a concrete backend based on available VRAM."""
    if requested != "auto":
        return requested
    info = gpu_info()
    if info and info["vram_total_mb"] >= min_vram_for_latentsync_mb:
        return "latentsync"
    return "wav2lip"
