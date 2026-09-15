#!/usr/bin/env python
"""Voice-cloning TTS worker. Runs inside envs/tts (Chatterbox). Invoked by the orchestrator.

Reads the script, splits it into sentence chunks, synthesises each chunk in the reference
voice, concatenates them with short pauses and writes a 16-bit mono WAV.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import fail, log, progress  # noqa: E402
from avatar_pipeline.text import chunk_text  # noqa: E402

MODELS = ("turbo", "nano", "english", "multilingual")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True, help="reference voice WAV (mono, ~10-15 s)")
    p.add_argument("--script", required=True, help="UTF-8 text file with the script")
    p.add_argument("--out", required=True, help="output WAV path")
    p.add_argument("--model", default="turbo", choices=MODELS)
    p.add_argument("--language", default="en", help="ISO 639-1 code (multilingual model only)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-chars", type=int, default=250)
    p.add_argument("--pause", type=float, default=0.3, help="seconds of silence between chunks")
    return p.parse_args()


def load_model(kind: str, device: str):
    log(f"[tts] loading Chatterbox '{kind}' on {device} (first run downloads the weights)")
    if kind in ("turbo", "nano"):
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        return ChatterboxTurboTTS.from_pretrained(device=device, nano=(kind == "nano"))
    if kind == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        try:
            return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        except TypeError:  # older package without the t3_model argument
            return ChatterboxMultilingualTTS.from_pretrained(device=device)
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=device)


def generate_chunk(model, kind: str, text: str, args: argparse.Namespace):
    kwargs = {"audio_prompt_path": args.ref}
    if kind == "multilingual":
        kwargs["language_id"] = args.language
    if kind in ("english", "multilingual"):
        kwargs.update(exaggeration=args.exaggeration, cfg_weight=args.cfg_weight,
                      temperature=args.temperature)
    else:
        kwargs.update(temperature=args.temperature)
    try:
        return model.generate(text, **kwargs)
    except TypeError as exc:
        # Signature drift between chatterbox releases: retry with the minimal call.
        log(f"[tts] generate() rejected tuning kwargs ({exc}); retrying with defaults")
        minimal = {"audio_prompt_path": args.ref}
        if kind == "multilingual":
            minimal["language_id"] = args.language
        return model.generate(text, **minimal)


def is_cuda_oom(exc: BaseException) -> bool:
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text or "cublas" in text


def synthesize(kind: str, device: str, chunks: list[str], args: argparse.Namespace):
    import torch
    model = load_model(kind, device)
    sr = int(model.sr)
    pause = torch.zeros(int(sr * max(args.pause, 0.0)))
    pieces = []
    total_chars = sum(len(c) for c in chunks)
    done_chars = 0
    for i, chunk in enumerate(chunks):
        t0 = time.time()
        wav = generate_chunk(model, kind, chunk, args)
        wav = wav.detach().float().cpu()
        if wav.dim() == 2:
            wav = wav[0]
        pieces.append(wav)
        pieces.append(pause)
        done_chars += len(chunk)
        progress(done_chars / max(total_chars, 1),
                 f"chunk {i + 1}/{len(chunks)}: {len(chunk)} chars, {wav.numel() / sr:.1f}s audio "
                 f"in {time.time() - t0:.1f}s")
    audio = torch.cat(pieces[:-1]) if pieces else torch.zeros(sr)
    return audio, sr


def write_wav(path: Path, audio, sr: int) -> None:
    import numpy as np
    pcm = (audio.clamp(-1.0, 1.0).numpy() * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()
    import torch

    text = Path(args.script).read_text(encoding="utf-8")
    chunks = chunk_text(text, max_chars=args.max_chars)
    if not chunks:
        fail("script is empty after normalisation")
    if not Path(args.ref).exists():
        fail(f"reference audio not found: {args.ref}")
    log(f"[tts] {len(chunks)} chunk(s), {sum(len(c) for c in chunks)} characters")

    if args.device == "auto":
        attempts = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    else:
        attempts = [args.device]
        if args.device == "cuda" and not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False in envs/tts")

    audio = sr = None
    used_device = None
    for index, device in enumerate(attempts):
        try:
            audio, sr = synthesize(args.model, device, chunks, args)
            used_device = device
            break
        except Exception as exc:  # noqa: BLE001 - we want to inspect and maybe retry
            last = index == len(attempts) - 1
            if device == "cuda" and not last and is_cuda_oom(exc):
                log(f"[tts] CUDA failure on {device} ({str(exc).splitlines()[0][:160]}); retrying on CPU")
                torch.cuda.empty_cache()
                continue
            raise
    if audio is None:
        fail("synthesis failed on every device")

    out = Path(args.out)
    write_wav(out, audio, sr)
    seconds = audio.numel() / sr
    log(f"[tts] wrote {out} ({seconds:.1f}s @ {sr} Hz, device={used_device}, model={args.model})")
    out.with_suffix(".json").write_text(json.dumps({
        "model": args.model, "device": used_device, "sample_rate": sr,
        "seconds": round(seconds, 3), "chunks": chunks,
    }, indent=2), encoding="utf-8")
    progress(1.0, "done")


if __name__ == "__main__":
    main()
