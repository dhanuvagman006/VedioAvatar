from __future__ import annotations

from pathlib import Path

from ..media import find_binaries
from ..paths import REPO_ROOT, WORKERS_DIR, env_python, resolve
from .base import BackendError, LipSyncBackend, LogFn, ProgressFn
from .runner import run_worker

PATCH_MARKER = ".vedioavatar_patched"


class Wav2LipBackend(LipSyncBackend):
    """Wav2Lip (Rudrabha et al.). ~1-2 GB VRAM, works on any CUDA GPU incl. a 4 GB A2000."""

    name = "wav2lip"
    ENV = "wav2lip"

    def python(self) -> Path:
        return env_python(self.cfg.envs_dir, self.ENV)

    @property
    def repo(self) -> Path:
        return self.cfg.third_party_dir / "Wav2Lip"

    @property
    def checkpoint(self) -> Path:
        return resolve(self.cfg.get("lipsync.wav2lip.checkpoint", "wav2lip/wav2lip_gan.pth"), self.cfg.models_dir)

    @property
    def s3fd(self) -> Path:
        return self.repo / "face_detection" / "detection" / "sfd" / "s3fd.pth"

    def sync(self, video: Path, audio: Path, out_video: Path, workdir: Path,
             log: LogFn, progress: ProgressFn) -> Path:
        problems = self.check()
        if problems:
            raise BackendError("Wav2Lip is not ready:\n- " + "\n- ".join(problems))
        w = self.cfg.section("lipsync.wav2lip")
        pads = [str(int(p)) for p in (w.get("pads") or [0, 10, 0, 0])]
        ffmpeg, _ = find_binaries(self.cfg.get("ffmpeg.binary") or None, self.cfg.get("ffmpeg.ffprobe") or None)
        args = [
            "--repo", self.repo, "--checkpoint", self.checkpoint,
            "--video", video, "--audio", audio, "--out", out_video,
            "--pads", *pads,
            "--resize-factor", str(int(w.get("resize_factor", 1))),
            "--face-det-batch-size", str(int(w.get("face_det_batch_size", 4))),
            "--batch-size", str(int(w.get("batch_size", 32))),
            "--ffmpeg-dir", Path(ffmpeg).parent,
        ]
        if w.get("nosmooth"):
            args.append("--nosmooth")
        run_worker(self.python(), WORKERS_DIR / "wav2lip_worker.py", args, cwd=REPO_ROOT,
                   log=log, progress=progress, timeout=self.cfg.get("worker_timeout_seconds"))
        if not out_video.exists():
            raise BackendError(f"Wav2Lip worker finished but did not write {out_video}")
        return out_video

    def check(self) -> list[str]:
        problems = []
        if not self.python().exists():
            problems.append(f"Wav2Lip environment missing ({self.python()}). Run: python scripts/setup_envs.py")
        if not (self.repo / "inference.py").exists():
            problems.append(f"Wav2Lip repo missing at {self.repo}. Run: python scripts/setup_envs.py")
        elif not (self.repo / PATCH_MARKER).exists():
            problems.append("Wav2Lip repo is not patched. Run: python scripts/patch_wav2lip.py")
        if not self.checkpoint.exists():
            problems.append(f"Checkpoint missing: {self.checkpoint}. Run: python scripts/download_models.py --wav2lip")
        if not self.s3fd.exists():
            problems.append(f"Face detector weights missing: {self.s3fd}. Run: python scripts/download_models.py --wav2lip")
        return problems
