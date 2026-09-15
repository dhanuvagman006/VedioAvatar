"""End-to-end orchestration: source clip + script -> lip-synced video in the cloned voice."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .backends import BackendError, get_lipsync_backend, get_tts_backend
from .config import Config
from .gpu import choose_lipsync_backend, gpu_info
from .media import FFmpeg, MediaError
from .text import normalize_script

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str, str], None]  # (overall percent 0..100, stage, message)

# (stage name, weight in percent)
STAGES: list[tuple[str, float]] = [
    ("probe", 2), ("reference", 5), ("driver", 6), ("tts", 35),
    ("extend", 5), ("lipsync", 42), ("finalize", 5),
]

# Flat option names accepted from the CLI / API, mapped to config keys.
OPTION_KEYS = {
    "tts_backend": "tts.backend",
    "tts_model": "tts.model",
    "language": "tts.language",
    "tts_device": "tts.device",
    "exaggeration": "tts.exaggeration",
    "cfg_weight": "tts.cfg_weight",
    "temperature": "tts.temperature",
    "lipsync_backend": "lipsync.backend",
    "max_height": "preprocess.max_height",
    "reference_seconds": "preprocess.reference_seconds",
    "denoise": "preprocess.denoise",
}


class PipelineError(RuntimeError):
    pass


def options_to_overrides(options: dict[str, Any] | None) -> dict[str, Any]:
    """Turn flat CLI/API options into the nested config shape (unknown keys are ignored)."""
    overrides: dict[str, Any] = {}
    for key, value in (options or {}).items():
        if value is None or value == "":
            continue
        if key in ("tts", "lipsync", "preprocess", "output") and isinstance(value, dict):
            overrides.setdefault(key, {}).update(value)
            continue
        dotted = OPTION_KEYS.get(key)
        if not dotted:
            continue
        node = overrides
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


@dataclass
class PipelineResult:
    output: Path
    workdir: Path
    manifest: dict[str, Any]


class Pipeline:
    def __init__(self, cfg: Config, log: LogFn | None = None, progress: ProgressFn | None = None):
        self.cfg = cfg
        self.log: LogFn = log or (lambda msg: print(msg, flush=True))
        self.progress: ProgressFn = progress or (lambda pct, stage, msg: None)

    # Progress bookkeeping ------------------------------------------------------
    def _report(self, stage: str, fraction: float, message: str = "") -> None:
        base = 0.0
        for name, weight in STAGES:
            if name == stage:
                self.progress(min(100.0, base + weight * max(0.0, min(1.0, fraction))), stage, message)
                return
            base += weight
        self.progress(base, stage, message)

    def resolve_backends(self, cfg: Config) -> tuple[str, str]:
        tts_name = str(cfg.get("tts.backend", "chatterbox"))
        lipsync_name = choose_lipsync_backend(
            str(cfg.get("lipsync.backend", "auto")),
            int(cfg.get("lipsync.auto_latentsync_min_vram_mb", 9000)),
        )
        return tts_name, lipsync_name

    def preflight(self, cfg: Config) -> tuple[str, str]:
        """Resolve backends and fail fast with actionable messages if they are not installed."""
        tts_name, lipsync_name = self.resolve_backends(cfg)
        problems = get_tts_backend(tts_name, cfg).check() + get_lipsync_backend(lipsync_name, cfg).check()
        if problems:
            raise PipelineError("Environment is not ready:\n- " + "\n- ".join(problems))
        return tts_name, lipsync_name

    # Main entry point -----------------------------------------------------------
    def run(self, video: Path, script_text: str, out_path: Path, workdir: Path,
            options: dict[str, Any] | None = None, ref_audio: Path | None = None) -> PipelineResult:
        cfg = self.cfg.with_overrides(options_to_overrides(options))
        video, out_path, workdir = Path(video), Path(out_path), Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}
        manifest: dict[str, Any] = {
            "version": __version__,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "gpu": gpu_info(),
            "input_video": str(video),
            "ref_audio": str(ref_audio) if ref_audio else None,
            "options": options or {},
            "stages": {},
        }

        def stage(name: str):
            self._report(name, 0.0, f"{name} started")
            timings[name] = time.time()
            self.log(f"== stage: {name}")

        def done(name: str, **info: Any):
            timings[name] = round(time.time() - timings[name], 2)
            manifest["stages"][name] = {"seconds": timings[name], **info}
            self._report(name, 1.0, f"{name} done in {timings[name]}s")

        try:
            ff = FFmpeg(cfg.get("ffmpeg.binary") or None, cfg.get("ffmpeg.ffprobe") or None, log=self.log)
            pre = cfg.section("preprocess")
            fps = int(pre.get("fps", 25))

            script = normalize_script(script_text)
            if not script:
                raise PipelineError("The script is empty.")
            script_path = workdir / "script.txt"
            script_path.write_text(script, encoding="utf-8")

            tts_name, lipsync_name = self.preflight(cfg)
            manifest["tts_backend"], manifest["lipsync_backend"] = tts_name, lipsync_name
            self.log(f"backends: tts={tts_name} ({cfg.get('tts.model')}), lipsync={lipsync_name}")

            # 1. Probe -------------------------------------------------------------
            stage("probe")
            info = ff.probe(video)
            if not info.has_video:
                raise PipelineError(f"{video.name} has no video stream.")
            if info.duration < 2.0:
                raise PipelineError(f"{video.name} is only {info.duration:.1f}s long; use at least a few seconds.")
            if not info.has_audio and ref_audio is None:
                raise PipelineError(f"{video.name} has no audio track. Upload a clip with the person's voice "
                                    "or supply a separate reference audio file.")
            self.log(f"input: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f}s, "
                     f"audio={'yes' if info.has_audio else 'no'}")
            done("probe", duration=info.duration, width=info.width, height=info.height, fps=info.fps)

            # 2. Reference voice ----------------------------------------------------
            stage("reference")
            ref_wav = workdir / "reference.wav"
            ff.extract_reference_audio(
                Path(ref_audio) if ref_audio else video, ref_wav,
                float(pre.get("reference_seconds", 12)), sample_rate=24000,
                loudnorm=bool(pre.get("loudnorm", True)), denoise=bool(pre.get("denoise", False)),
            )
            ref_seconds = ff.duration(ref_wav)
            if ref_seconds < 1.5:
                raise PipelineError(f"Reference audio is only {ref_seconds:.1f}s after trimming silence; "
                                    "the clip needs several seconds of clear speech.")
            done("reference", seconds=ref_seconds)

            # 3. Driver video (silent, 25 fps, capped height) ---------------------------
            stage("driver")
            driver_src = workdir / "driver_source.mp4"
            ff.prepare_driver_video(video, driver_src, max_height=int(pre.get("max_height", 720)),
                                    fps=fps, max_seconds=float(pre.get("max_source_seconds", 60)))
            done("driver", seconds=ff.duration(driver_src))

            # 4. Voice-cloned speech --------------------------------------------------
            stage("tts")
            speech = workdir / "speech.wav"
            get_tts_backend(tts_name, cfg).synthesize(
                ref_wav, script_path, speech, workdir, self.log,
                lambda f, m: self._report("tts", f, m))
            speech_seconds = ff.duration(speech)
            if speech_seconds <= 0.1:
                raise PipelineError("TTS produced no audio.")
            done("tts", seconds=speech_seconds, chars=len(script))

            # 5. Extend / trim the driver to the speech length ---------------------------
            stage("extend")
            driver = workdir / "driver.mp4"
            ff.extend_video_pingpong(driver_src, driver, speech_seconds + 0.25, fps=fps)
            done("extend", seconds=ff.duration(driver))

            # 6. Lip-sync -----------------------------------------------------------
            stage("lipsync")
            synced = workdir / "lipsync_raw.mp4"
            get_lipsync_backend(lipsync_name, cfg).sync(
                driver, speech, synced, workdir, self.log,
                lambda f, m: self._report("lipsync", f, m))
            done("lipsync")

            # 7. Final encode -------------------------------------------------------
            stage("finalize")
            out_cfg = cfg.section("output")
            ff.mux(synced, speech, out_path, crf=int(out_cfg.get("crf", 18)),
                   preset=str(out_cfg.get("preset", "medium")),
                   audio_bitrate=str(out_cfg.get("audio_bitrate", "192k")))
            final = ff.probe(out_path)
            done("finalize", seconds=final.duration, width=final.width, height=final.height)
        except (MediaError, BackendError) as exc:
            raise PipelineError(str(exc)) from exc
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        manifest["output"] = str(out_path)
        (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        self.log(f"output written to {out_path}")
        return PipelineResult(output=out_path, workdir=workdir, manifest=manifest)
