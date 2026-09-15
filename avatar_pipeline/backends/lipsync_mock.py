from __future__ import annotations

from pathlib import Path

from ..media import FFmpeg
from .base import LipSyncBackend, LogFn, ProgressFn


class MockLipSyncBackend(LipSyncBackend):
    """Copies the driver video unchanged (audio muxed in). For plumbing tests without a GPU."""

    name = "mock"

    def sync(self, video: Path, audio: Path, out_video: Path, workdir: Path,
             log: LogFn, progress: ProgressFn) -> Path:
        ff = FFmpeg(self.cfg.get("ffmpeg.binary") or None, self.cfg.get("ffmpeg.ffprobe") or None, log=log)
        log("[mock-lipsync] passing the driver video through without modification")
        ff.mux(video, audio, out_video, copy_video=True)
        progress(1.0, "done")
        return out_video
