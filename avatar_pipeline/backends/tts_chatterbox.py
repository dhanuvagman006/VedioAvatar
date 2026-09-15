from __future__ import annotations

from pathlib import Path

from ..paths import REPO_ROOT, WORKERS_DIR, env_python
from .base import BackendError, LogFn, ProgressFn, TTSBackend
from .runner import run_worker

MODELS = ("turbo", "nano", "english", "multilingual")


class ChatterboxBackend(TTSBackend):
    """Zero-shot voice cloning with Resemble AI's Chatterbox (MIT), run inside envs/tts."""

    name = "chatterbox"
    ENV = "tts"

    def python(self) -> Path:
        return env_python(self.cfg.envs_dir, self.ENV)

    def synthesize(self, ref_wav: Path, script_path: Path, out_wav: Path, workdir: Path,
                   log: LogFn, progress: ProgressFn) -> Path:
        tts = self.cfg.section("tts")
        model = str(tts.get("model", "turbo")).lower()
        if model not in MODELS:
            raise BackendError(f"Unknown tts.model '{model}'. Choose from: {', '.join(MODELS)}")
        args = [
            "--ref", ref_wav, "--script", script_path, "--out", out_wav,
            "--model", model,
            "--language", str(tts.get("language", "en")),
            "--device", str(tts.get("device", "auto")),
            "--exaggeration", str(tts.get("exaggeration", 0.5)),
            "--cfg-weight", str(tts.get("cfg_weight", 0.5)),
            "--temperature", str(tts.get("temperature", 0.8)),
            "--max-chars", str(tts.get("max_chunk_chars", 250)),
            "--pause", str(tts.get("pause_seconds", 0.3)),
        ]
        run_worker(self.python(), WORKERS_DIR / "tts_chatterbox_worker.py", args, cwd=REPO_ROOT,
                   log=log, progress=progress, timeout=self.cfg.get("worker_timeout_seconds"))
        if not out_wav.exists():
            raise BackendError(f"TTS worker finished but did not write {out_wav}")
        return out_wav

    def check(self) -> list[str]:
        py = self.python()
        if not py.exists():
            return [f"TTS environment missing ({py}). Run: python scripts/setup_envs.py"]
        return []
