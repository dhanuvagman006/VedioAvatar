"""Command line interface: `avatar run|serve|doctor|chunks`."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_config
from .text import chunk_text


def _add_common_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to config.yaml (default: ./config.yaml or $AVATAR_CONFIG)")


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline, PipelineError

    cfg = load_config(args.config)
    if args.text:
        script = args.text
    else:
        script = Path(args.script).read_text(encoding="utf-8")
    out = Path(args.out)
    workdir = Path(args.workdir) if args.workdir else cfg.data_dir / "cli" / (out.stem + "_" + time.strftime("%Y%m%d-%H%M%S"))
    options = {
        "tts_backend": args.tts_backend, "tts_model": args.tts_model, "language": args.language,
        "tts_device": args.tts_device, "lipsync_backend": args.lipsync_backend,
        "exaggeration": args.exaggeration, "cfg_weight": args.cfg_weight, "temperature": args.temperature,
        "denoise": args.denoise or None,
    }
    started = time.time()

    def progress(pct: float, stage: str, message: str) -> None:
        print(f"[{pct:5.1f}%] {stage}: {message}", file=sys.stderr, flush=True)

    try:
        result = Pipeline(cfg, progress=progress).run(
            Path(args.video), script, out, workdir, options,
            ref_audio=Path(args.ref_audio) if args.ref_audio else None)
    except PipelineError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone in {time.time() - started:.0f}s -> {result.output}\nWork files: {result.workdir}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    cfg = load_config(args.config)
    host = args.host or cfg.get("server.host", "0.0.0.0")
    port = int(args.port or cfg.get("server.port", 8000))
    print(f"VedioAvatar {__version__} listening on http://{host}:{port}  (open http://localhost:{port} in a browser)")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import format_checks, full_checks

    checks = full_checks(load_config(args.config))
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks if c.name not in ("lipsync:latentsync", "env:latentsync")) else 1


def cmd_chunks(args: argparse.Namespace) -> int:
    text = Path(args.script).read_text(encoding="utf-8")
    for i, chunk in enumerate(chunk_text(text, max_chars=args.max_chars), 1):
        print(f"--- chunk {i} ({len(chunk)} chars)\n{chunk}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avatar", description="VedioAvatar pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate a talking video from a clip and a script")
    run.add_argument("--video", required=True, help="source clip with the person's face and voice")
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--script", help="UTF-8 text file to read out")
    group.add_argument("--text", help="script given inline")
    run.add_argument("--out", required=True, help="output .mp4 path")
    run.add_argument("--workdir", help="where intermediate files go (default: data/cli/<name>)")
    run.add_argument("--ref-audio", help="use this audio file as the voice reference instead of the video's track")
    run.add_argument("--tts-backend", choices=("chatterbox", "mock"))
    run.add_argument("--tts-model", choices=("turbo", "nano", "english", "multilingual"))
    run.add_argument("--tts-device", choices=("auto", "cuda", "cpu"))
    run.add_argument("--language", help="ISO 639-1 code for the multilingual TTS model")
    run.add_argument("--lipsync-backend", choices=("auto", "wav2lip", "latentsync", "mock"))
    run.add_argument("--exaggeration", type=float)
    run.add_argument("--cfg-weight", type=float)
    run.add_argument("--temperature", type=float)
    run.add_argument("--denoise", action="store_true", help="denoise the voice reference")
    _add_common_options(run)
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="start the web UI / REST API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    _add_common_options(serve)
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="check ffmpeg, GPU, environments and model weights")
    _add_common_options(doctor)
    doctor.set_defaults(func=cmd_doctor)

    chunks = sub.add_parser("chunks", help="show how a script is split for the TTS model")
    chunks.add_argument("--script", required=True)
    chunks.add_argument("--max-chars", type=int, default=250)
    chunks.set_defaults(func=cmd_chunks)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
