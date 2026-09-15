from __future__ import annotations

from pathlib import Path

from ..media import find_binaries
from ..paths import REPO_ROOT, WORKERS_DIR, env_python
from .base import BackendError, LipSyncBackend, LogFn, ProgressFn
from .runner import run_worker

VERSIONS = {
    "1.5": {"hf_repo": "ByteDance/LatentSync-1.5", "config": "configs/unet/stage2.yaml", "min_vram_mb": 8000},
    "1.6": {"hf_repo": "ByteDance/LatentSync-1.6", "config": "configs/unet/stage2_512.yaml", "min_vram_mb": 18000},
}


class LatentSyncBackend(LipSyncBackend):
    """ByteDance LatentSync (Apache-2.0). Much better quality than Wav2Lip but needs >= 8 GB VRAM."""

    name = "latentsync"
    ENV = "latentsync"

    def python(self) -> Path:
        return env_python(self.cfg.envs_dir, self.ENV)

    @property
    def repo(self) -> Path:
        return self.cfg.third_party_dir / "LatentSync"

    @property
    def version(self) -> str:
        v = str(self.cfg.get("lipsync.latentsync.version", "1.5"))
        if v not in VERSIONS:
            raise BackendError(f"Unsupported lipsync.latentsync.version '{v}'. Choose from: {', '.join(VERSIONS)}")
        return v

    @property
    def unet_checkpoint(self) -> Path:
        return self.repo / "checkpoints" / "latentsync_unet.pt"

    @property
    def whisper_checkpoint(self) -> Path:
        return self.repo / "checkpoints" / "whisper" / "tiny.pt"

    def sync(self, video: Path, audio: Path, out_video: Path, workdir: Path,
             log: LogFn, progress: ProgressFn) -> Path:
        problems = self.check()
        if problems:
            raise BackendError("LatentSync is not ready:\n- " + "\n- ".join(problems))
        ls = self.cfg.section("lipsync.latentsync")
        ffmpeg, _ = find_binaries(self.cfg.get("ffmpeg.binary") or None, self.cfg.get("ffmpeg.ffprobe") or None)
        args = [
            "--repo", self.repo,
            "--config", VERSIONS[self.version]["config"],
            "--checkpoint", "checkpoints/latentsync_unet.pt",
            "--video", video, "--audio", audio, "--out", out_video,
            "--steps", str(int(ls.get("inference_steps", 20))),
            "--guidance", str(float(ls.get("guidance_scale", 1.5))),
            "--seed", str(int(ls.get("seed", 1247))),
            "--ffmpeg-dir", Path(ffmpeg).parent,
        ]
        if ls.get("enable_deepcache", True):
            args.append("--deepcache")
        run_worker(self.python(), WORKERS_DIR / "latentsync_worker.py", args, cwd=REPO_ROOT,
                   log=log, progress=progress, timeout=self.cfg.get("worker_timeout_seconds"))
        if not out_video.exists():
            raise BackendError(f"LatentSync worker finished but did not write {out_video}")
        return out_video

    def check(self) -> list[str]:
        problems = []
        if not self.python().exists():
            problems.append(f"LatentSync environment missing ({self.python()}). Run: python scripts/setup_envs.py --lipsync latentsync")
        if not (self.repo / "scripts" / "inference.py").exists():
            problems.append(f"LatentSync repo missing at {self.repo}. Run: python scripts/setup_envs.py --lipsync latentsync")
        if not self.unet_checkpoint.exists() or not self.whisper_checkpoint.exists():
            problems.append("LatentSync checkpoints missing. Run: python scripts/download_models.py --latentsync")
        return problems
