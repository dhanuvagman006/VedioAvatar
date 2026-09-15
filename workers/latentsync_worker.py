#!/usr/bin/env python
"""Lip-sync worker for LatentSync. Runs inside envs/latentsync. Invoked by the orchestrator.

Wraps `python -m scripts.inference` from the upstream repo (paths inside the repo are
hard-coded relative to its root, e.g. checkpoints/whisper/tiny.pt, so we run from there).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import env_with_ffmpeg, fail, log, progress, run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--config", default="configs/unet/stage2.yaml", help="unet config, relative to repo")
    p.add_argument("--checkpoint", default="checkpoints/latentsync_unet.pt", help="relative to repo or absolute")
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=1247)
    p.add_argument("--deepcache", action="store_true")
    p.add_argument("--ffmpeg-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "scripts" / "inference.py").exists():
        fail(f"LatentSync repo not found at {repo}")
    for required in (args.checkpoint, "checkpoints/whisper/tiny.pt", args.config):
        if not (repo / required).exists() and not Path(required).exists():
            fail(f"missing {required} under {repo} (run scripts/download_models.py --latentsync)")

    temp = repo / "temp" / f"job_{os.getpid()}"
    temp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.video, temp / "input_video.mp4")
    shutil.copyfile(args.audio, temp / "input_audio.wav")
    rel = temp.relative_to(repo).as_posix()

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path", args.config,
        "--inference_ckpt_path", args.checkpoint,
        "--inference_steps", str(args.steps),
        "--guidance_scale", str(args.guidance),
        "--seed", str(args.seed),
        "--video_path", f"{rel}/input_video.mp4",
        "--audio_path", f"{rel}/input_audio.wav",
        "--video_out_path", f"{rel}/output.mp4",
        "--temp_dir", f"{rel}/work",
    ]
    if args.deepcache:
        cmd.append("--enable_deepcache")

    progress(0.02, "running LatentSync inference")
    rc = run(cmd, cwd=repo, env=env_with_ffmpeg(args.ffmpeg_dir))
    produced = temp / "output.mp4"
    if rc != 0 or not produced.exists():
        fail(f"LatentSync inference failed (exit code {rc}). Check VRAM (1.5 needs ~8 GB) and that a "
             "face is visible in every frame.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(out))
    shutil.rmtree(temp, ignore_errors=True)
    log(f"[latentsync] wrote {out}")
    progress(1.0, "done")


if __name__ == "__main__":
    main()
