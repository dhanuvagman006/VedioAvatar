#!/usr/bin/env python
"""Download model weights. Run from the orchestrator environment (needs huggingface_hub).

  python scripts/download_models.py --wav2lip                 # default, ~530 MB
  python scripts/download_models.py --latentsync [--latentsync-version 1.5]   # ~5.2 GB
  python scripts/download_models.py --tts turbo               # optional pre-fetch of Chatterbox weights

Chatterbox weights are otherwise fetched automatically on the first TTS run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avatar_pipeline.backends.lipsync_latentsync import VERSIONS  # noqa: E402
from avatar_pipeline.config import load_config  # noqa: E402

WAV2LIP_HF_REPO = "camenduru/Wav2Lip"
WAV2LIP_FILES = {"checkpoints/wav2lip_gan.pth": "wav2lip_gan.pth",
                 "checkpoints/s3fd-619a316812.pth": "s3fd.pth"}
CHATTERBOX_REPOS = {"turbo": "ResembleAI/chatterbox-turbo", "nano": "ResembleAI/chatterbox-nano",
                    "english": "ResembleAI/chatterbox", "multilingual": "ResembleAI/chatterbox"}


def download_wav2lip(models_dir: Path, third_party_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    target = models_dir / "wav2lip"
    target.mkdir(parents=True, exist_ok=True)
    for remote, local_name in WAV2LIP_FILES.items():
        dest = target / local_name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {WAV2LIP_HF_REPO}/{remote} ...")
        got = hf_hub_download(WAV2LIP_HF_REPO, remote, local_dir=models_dir / "_hf" / "wav2lip")
        shutil.move(got, dest)
        print(f"saved: {dest}")
    sfd_dir = third_party_dir / "Wav2Lip" / "face_detection" / "detection" / "sfd"
    if sfd_dir.exists():
        dest = sfd_dir / "s3fd.pth"
        if not dest.exists():
            shutil.copyfile(target / "s3fd.pth", dest)
            print(f"installed face detector: {dest}")
    else:
        print(f"note: Wav2Lip repo not cloned yet ({sfd_dir.parent.parent.parent}); "
              "re-run this script after scripts/setup_envs.py to install s3fd.pth")


def download_latentsync(third_party_dir: Path, version: str) -> None:
    from huggingface_hub import hf_hub_download

    spec = VERSIONS[version]
    ckpt_dir = third_party_dir / "LatentSync" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for remote in ("latentsync_unet.pt", "whisper/tiny.pt"):
        dest = ckpt_dir / remote
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {spec['hf_repo']}/{remote} ...")
        hf_hub_download(spec["hf_repo"], remote, local_dir=ckpt_dir)
        print(f"saved: {dest}")


def prefetch_chatterbox(model: str) -> None:
    from huggingface_hub import snapshot_download

    repo = CHATTERBOX_REPOS[model]
    print(f"pre-fetching {repo} into the Hugging Face cache ...")
    print(snapshot_download(repo))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav2lip", action="store_true")
    parser.add_argument("--latentsync", action="store_true")
    parser.add_argument("--latentsync-version", default=None, choices=sorted(VERSIONS))
    parser.add_argument("--tts", choices=sorted(CHATTERBOX_REPOS), help="pre-fetch Chatterbox weights")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not (args.wav2lip or args.latentsync or args.tts):
        args.wav2lip = True
    if args.wav2lip:
        download_wav2lip(cfg.models_dir, cfg.third_party_dir)
    if args.latentsync:
        download_latentsync(cfg.third_party_dir, args.latentsync_version or str(cfg.get("lipsync.latentsync.version", "1.5")))
    if args.tts:
        prefetch_chatterbox(args.tts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
