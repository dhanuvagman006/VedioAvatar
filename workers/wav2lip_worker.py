#!/usr/bin/env python
"""Lip-sync worker for Wav2Lip. Runs inside envs/wav2lip. Invoked by the orchestrator.

Stages the inputs under <repo>/temp with space-free relative names (Wav2Lip's own
ffmpeg call does not quote paths), runs the upstream inference.py, and moves the result.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import env_with_ffmpeg, fail, log, progress, run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0])
    p.add_argument("--resize-factor", type=int, default=1)
    p.add_argument("--face-det-batch-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--nosmooth", action="store_true")
    p.add_argument("--ffmpeg-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "inference.py").exists():
        fail(f"Wav2Lip repo not found at {repo}")
    s3fd = repo / "face_detection" / "detection" / "sfd" / "s3fd.pth"
    if not s3fd.exists():
        fail(f"missing face detector weights {s3fd} (run scripts/download_models.py --wav2lip)")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        fail(f"missing checkpoint {checkpoint}")

    temp = repo / "temp"
    temp.mkdir(exist_ok=True)
    for stale in ("result.avi", "temp.wav", "faulty_frame.jpg", "input_video.mp4", "input_audio.wav", "output.mp4"):
        try:
            (temp / stale).unlink()
        except FileNotFoundError:
            pass
    shutil.copyfile(args.video, temp / "input_video.mp4")
    shutil.copyfile(args.audio, temp / "input_audio.wav")

    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", str(checkpoint),
        "--face", "temp/input_video.mp4",
        "--audio", "temp/input_audio.wav",
        "--outfile", "temp/output.mp4",
        "--pads", *map(str, args.pads),
        "--resize_factor", str(args.resize_factor),
        "--face_det_batch_size", str(args.face_det_batch_size),
        "--wav2lip_batch_size", str(args.batch_size),
    ]
    if args.nosmooth:
        cmd.append("--nosmooth")

    progress(0.02, "running Wav2Lip inference")
    rc = run(cmd, cwd=repo, env=env_with_ffmpeg(args.ffmpeg_dir))
    produced = temp / "output.mp4"
    if rc != 0 or not produced.exists():
        hint = ""
        if (temp / "faulty_frame.jpg").exists():
            hint = (" Wav2Lip could not find a face in at least one frame (see "
                    f"{temp / 'faulty_frame.jpg'}). Keep the face fully visible for the whole clip.")
        fail(f"Wav2Lip inference failed (exit code {rc}).{hint}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(out))
    log(f"[wav2lip] wrote {out}")
    progress(1.0, "done")


if __name__ == "__main__":
    main()
