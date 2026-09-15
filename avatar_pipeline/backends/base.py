from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable

from ..config import Config

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]  # (fraction within stage 0..1, message)


class BackendError(RuntimeError):
    pass


class TTSBackend(abc.ABC):
    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def synthesize(self, ref_wav: Path, script_path: Path, out_wav: Path, workdir: Path,
                   log: LogFn, progress: ProgressFn) -> Path:
        """Speak the script in the reference voice; return the written WAV path."""

    def check(self) -> list[str]:
        """Return a list of setup problems (empty means ready)."""
        return []


class LipSyncBackend(abc.ABC):
    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abc.abstractmethod
    def sync(self, video: Path, audio: Path, out_video: Path, workdir: Path,
             log: LogFn, progress: ProgressFn) -> Path:
        """Re-render the mouth region of ``video`` to match ``audio``; return the output path."""

    def check(self) -> list[str]:
        return []


def _tts_registry() -> dict[str, type[TTSBackend]]:
    from .tts_chatterbox import ChatterboxBackend
    from .tts_mock import MockTTSBackend
    return {ChatterboxBackend.name: ChatterboxBackend, MockTTSBackend.name: MockTTSBackend}


def _lipsync_registry() -> dict[str, type[LipSyncBackend]]:
    from .lipsync_latentsync import LatentSyncBackend
    from .lipsync_mock import MockLipSyncBackend
    from .lipsync_wav2lip import Wav2LipBackend
    return {
        Wav2LipBackend.name: Wav2LipBackend,
        LatentSyncBackend.name: LatentSyncBackend,
        MockLipSyncBackend.name: MockLipSyncBackend,
    }


def get_tts_backend(name: str, cfg: Config) -> TTSBackend:
    registry = _tts_registry()
    if name not in registry:
        raise BackendError(f"Unknown TTS backend '{name}'. Choose from: {', '.join(registry)}")
    return registry[name](cfg)


def get_lipsync_backend(name: str, cfg: Config) -> LipSyncBackend:
    registry = _lipsync_registry()
    if name not in registry:
        raise BackendError(f"Unknown lip-sync backend '{name}'. Choose from: {', '.join(registry)}")
    return registry[name](cfg)


def list_backends() -> dict[str, list[str]]:
    return {"tts": list(_tts_registry()), "lipsync": list(_lipsync_registry())}
