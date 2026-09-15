#!/usr/bin/env python
"""Make the upstream Wav2Lip checkout run on a modern Python / torch / librosa stack.

Idempotent. Writes a marker file (.vedioavatar_patched) when done.

Patches applied:
  * audio.py            librosa.filters.mel() keyword-only arguments (librosa >= 0.10)
  * inference.py        torch.load(..., weights_only=False) (torch >= 2.6 default changed)
  * sfd_detector.py     same torch.load change for the S3FD face detector
  * *.py                np.float / np.int / np.bool aliases removed in NumPy >= 1.24
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = ".vedioavatar_patched"


def patch_file(path: Path, replacements: list[tuple[str, str]], regex: bool = False) -> int:
    if not path.exists():
        print(f"  skip (missing): {path}")
        return 0
    text = original = path.read_text(encoding="utf-8")
    for old, new in replacements:
        text = re.sub(old, new, text) if regex else text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"  patched: {path}")
        return 1
    return 0


def patch_torch_load(path: Path) -> int:
    """Add weights_only=False to torch.load(...) calls that do not set it already."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")

    def repl(match: re.Match) -> str:
        inner = match.group(1)
        if "weights_only" in inner:
            return match.group(0)
        return f"torch.load({inner}, weights_only=False)"

    new = re.sub(r"torch\.load\(((?:[^()]|\([^()]*\))*)\)", repl, text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        print(f"  patched torch.load: {path}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default="third_party/Wav2Lip")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "inference.py").exists():
        print(f"Wav2Lip repo not found at {repo}", file=sys.stderr)
        return 1

    print(f"Patching {repo}")
    changed = 0
    changed += patch_file(repo / "audio.py", [
        ("librosa.filters.mel(hp.sample_rate, hp.n_fft,", "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,"),
    ])
    changed += patch_torch_load(repo / "inference.py")
    changed += patch_torch_load(repo / "face_detection" / "detection" / "sfd" / "sfd_detector.py")
    changed += patch_torch_load(repo / "face_detection" / "api.py")
    for py in repo.rglob("*.py"):
        changed += patch_file(py, [(r"\bnp\.(float|int|bool)\b(?![0-9_])", r"\1")], regex=True)
    (repo / MARKER).write_text("patched by scripts/patch_wav2lip.py\n", encoding="utf-8")
    print(f"Done ({changed} file(s) changed). Marker written: {repo / MARKER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
