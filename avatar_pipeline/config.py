from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT, resolve

DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "models_dir": "models",
        "third_party_dir": "third_party",
        "envs_dir": "envs",
    },
    "ffmpeg": {"binary": "", "ffprobe": ""},
    "preprocess": {
        "max_height": 720,
        "fps": 25,
        "max_source_seconds": 60,
        "reference_seconds": 12,
        "loudnorm": True,
        "denoise": False,
    },
    "tts": {
        "backend": "chatterbox",
        "model": "turbo",
        "language": "en",
        "device": "auto",
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "max_chunk_chars": 250,
        "pause_seconds": 0.3,
    },
    "lipsync": {
        "backend": "auto",
        "auto_latentsync_min_vram_mb": 9000,
        "wav2lip": {
            "checkpoint": "wav2lip/wav2lip_gan.pth",
            "pads": [0, 10, 0, 0],
            "resize_factor": 1,
            "face_det_batch_size": 4,
            "batch_size": 32,
            "nosmooth": False,
        },
        "latentsync": {
            "version": "1.5",
            "inference_steps": 20,
            "guidance_scale": 1.5,
            "seed": 1247,
            "enable_deepcache": True,
        },
    },
    "output": {"crf": 18, "preset": "medium", "audio_bitrate": "192k"},
    "server": {"host": "0.0.0.0", "port": 8000, "max_upload_mb": 500},
    "worker_timeout_seconds": 7200,
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Dictionary-backed configuration with dotted-path access and resolved paths."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self.data = data
        self.source = source

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    # Resolved directories -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return resolve(self.get("paths.data_dir", "data"))

    @property
    def models_dir(self) -> Path:
        return resolve(self.get("paths.models_dir", "models"))

    @property
    def third_party_dir(self) -> Path:
        return resolve(self.get("paths.third_party_dir", "third_party"))

    @property
    def envs_dir(self) -> Path:
        return resolve(self.get("paths.envs_dir", "envs"))

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Config":
        """Return a copy with a nested override dict merged in."""
        if not overrides:
            return Config(copy.deepcopy(self.data), self.source)
        return Config(_deep_merge(self.data, overrides), self.source)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml (or ``AVATAR_CONFIG``) merged over built-in defaults."""
    candidate = path or os.environ.get("AVATAR_CONFIG") or (REPO_ROOT / "config.yaml")
    cfg_path = Path(candidate)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {cfg_path} must contain a mapping at the top level")
        data = loaded
    return Config(_deep_merge(DEFAULTS, data), cfg_path if cfg_path.exists() else None)
