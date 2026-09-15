from __future__ import annotations

from pathlib import Path

from ..media import FFmpeg
from ..text import estimate_speech_seconds
from .base import LogFn, ProgressFn, TTSBackend


class MockTTSBackend(TTSBackend):
    """Produces a tone whose length matches the script. For plumbing tests without a GPU."""

    name = "mock"

    def synthesize(self, ref_wav: Path, script_path: Path, out_wav: Path, workdir: Path,
                   log: LogFn, progress: ProgressFn) -> Path:
        text = script_path.read_text(encoding="utf-8")
        seconds = estimate_speech_seconds(text)
        ff = FFmpeg(self.cfg.get("ffmpeg.binary") or None, self.cfg.get("ffmpeg.ffprobe") or None, log=log)
        log(f"[mock-tts] {len(text.split())} words -> {seconds:.1f}s of placeholder audio")
        ff.tone(out_wav, seconds)
        progress(1.0, "done")
        return out_wav
