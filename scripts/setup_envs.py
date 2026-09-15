#!/usr/bin/env python
"""One-shot setup: per-model virtual environments, upstream repos, patches and weights.

Run from the orchestrator environment (after `pip install -r requirements.txt`):

  python scripts/setup_envs.py                       # TTS + Wav2Lip (fits a 4 GB GPU)
  python scripts/setup_envs.py --lipsync both        # also LatentSync (needs >= 8 GB VRAM, Linux/WSL)
  python scripts/setup_envs.py --cuda cu121          # pick the CUDA wheel flavour (default cu124)
  python scripts/setup_envs.py --skip-models         # environments only

Each model gets its own interpreter under envs/ because their dependency pins conflict:
  envs/tts         Python 3.11, torch 2.6.0, chatterbox-tts
  envs/wav2lip     Python 3.10, torch 2.6.0, librosa 0.10
  envs/latentsync  Python 3.10, torch 2.5.1, LatentSync requirements (optional)
Python versions are fetched automatically by `uv` if they are not installed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from avatar_pipeline.config import load_config  # noqa: E402
from avatar_pipeline.paths import env_python  # noqa: E402

WAV2LIP_GIT = "https://github.com/Rudrabha/Wav2Lip.git"
LATENTSYNC_GIT = "https://github.com/bytedance/LatentSync.git"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{cuda}"


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def uv_command() -> list[str]:
    exe = shutil.which("uv")
    if exe:
        return [exe]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True, text=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "uv"]
    raise SystemExit("uv not found. Install the orchestrator requirements first: pip install -r requirements.txt")


class Setup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_config(args.config)
        self.uv = uv_command()
        self.envs = self.cfg.envs_dir
        self.third_party = self.cfg.third_party_dir
        self.envs.mkdir(parents=True, exist_ok=True)
        self.third_party.mkdir(parents=True, exist_ok=True)
        self.torch_index = PYTORCH_INDEX.format(cuda=args.cuda)

    def create_env(self, name: str, python_version: str) -> Path:
        py = env_python(self.envs, name)
        if py.exists() and not self.args.recreate:
            print(f"env '{name}' exists: {py}")
            return py
        if py.exists():
            shutil.rmtree(self.envs / name)
        sh([*self.uv, "venv", "--python", python_version, str(self.envs / name)])
        return py

    def pip(self, py: Path, *packages: str, index_url: str | None = None,
            extra_index_url: str | None = None, best_match: bool = False) -> None:
        cmd = [*self.uv, "pip", "install", "--python", str(py)]
        if index_url:
            cmd += ["--index-url", index_url]
        if extra_index_url:
            cmd += ["--extra-index-url", extra_index_url]
        if best_match:
            cmd += ["--index-strategy", "unsafe-best-match"]
        sh(cmd + list(packages))

    def clone(self, url: str, dest: Path) -> None:
        if (dest / ".git").exists():
            print(f"repo exists: {dest}")
            return
        if not shutil.which("git"):
            raise SystemExit("git is required to clone the upstream repos (https://git-scm.com/downloads)")
        sh(["git", "clone", "--depth", "1", url, str(dest)])

    # Environments -----------------------------------------------------------
    def setup_tts(self) -> None:
        print("\n### TTS environment (Chatterbox)")
        py = self.create_env("tts", self.args.python_tts)
        self.pip(py, "torch==2.6.0", "torchaudio==2.6.0", index_url=self.torch_index)
        self.pip(py, "chatterbox-tts")
        # If the resolver swapped in PyPI's CPU-only torch (Windows), put the CUDA build back.
        self.pip(py, "torch==2.6.0", "torchaudio==2.6.0", index_url=self.torch_index)

    def setup_wav2lip(self) -> None:
        print("\n### Wav2Lip environment")
        py = self.create_env("wav2lip", self.args.python_lipsync)
        self.pip(py, "torch==2.6.0", "torchvision==0.21.0", index_url=self.torch_index)
        self.pip(py, "numpy<2", "opencv-python", "librosa==0.10.2.post1", "numba", "scipy", "tqdm")
        self.pip(py, "torch==2.6.0", "torchvision==0.21.0", index_url=self.torch_index)
        self.clone(WAV2LIP_GIT, self.third_party / "Wav2Lip")
        sh([sys.executable, str(REPO_ROOT / "scripts" / "patch_wav2lip.py"), str(self.third_party / "Wav2Lip")])
        (self.third_party / "Wav2Lip" / "temp").mkdir(exist_ok=True)

    def setup_latentsync(self) -> None:
        print("\n### LatentSync environment (optional, >= 8 GB VRAM)")
        if os.name == "nt":
            print("WARNING: LatentSync's dependencies (insightface, decord, onnxruntime-gpu) are much easier "
                  "to install on Linux or WSL2 than on native Windows. Continuing anyway.")
        py = self.create_env("latentsync", self.args.python_lipsync)
        self.clone(LATENTSYNC_GIT, self.third_party / "LatentSync")
        req = self.third_party / "LatentSync" / "requirements.txt"
        self.pip(py, "-r", str(req), best_match=True)
        self.pip(py, "huggingface_hub")

    def download_models(self) -> None:
        print("\n### Model weights")
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "download_models.py")]
        if self.args.lipsync in ("wav2lip", "both"):
            cmd.append("--wav2lip")
        if self.args.lipsync in ("latentsync", "both"):
            cmd.append("--latentsync")
        if self.args.prefetch_tts:
            cmd += ["--tts", self.args.prefetch_tts]
        sh(cmd)

    def run(self) -> None:
        if not self.args.skip_envs:
            self.setup_tts()
            if self.args.lipsync in ("wav2lip", "both"):
                self.setup_wav2lip()
            if self.args.lipsync in ("latentsync", "both"):
                self.setup_latentsync()
        if not self.args.skip_models:
            self.download_models()
        print("\nSetup finished. Next:\n"
              "  python -m avatar_pipeline.cli doctor\n"
              "  python -m avatar_pipeline.cli run --video examples/sample_face.mp4 "
              "--script examples/sample_script.txt --out out.mp4\n"
              "  python -m avatar_pipeline.cli serve   # then open http://localhost:8000")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lipsync", choices=("wav2lip", "latentsync", "both"), default="wav2lip")
    parser.add_argument("--cuda", default="cu124", help="PyTorch wheel flavour: cu118, cu121, cu124, cu126, cpu")
    parser.add_argument("--python-tts", default="3.11")
    parser.add_argument("--python-lipsync", default="3.10")
    parser.add_argument("--prefetch-tts", choices=("turbo", "nano", "english", "multilingual"))
    parser.add_argument("--skip-envs", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild existing environments")
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        Setup(args).run()
    except subprocess.CalledProcessError as exc:
        print(f"\nsetup step failed (exit {exc.returncode}): {' '.join(map(str, exc.cmd))}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
