"""Thin, dependency-free wrappers around ffmpeg / ffprobe used by the orchestrator."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

LogFn = Callable[[str], None]


class MediaError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    audio_sample_rate: int = 0


def _which_static_ffmpeg() -> tuple[str | None, str | None]:
    """Fallback: the ``static-ffmpeg`` package ships ffmpeg+ffprobe binaries."""
    try:
        import static_ffmpeg  # type: ignore
    except ImportError:
        return None, None
    try:
        static_ffmpeg.add_paths()  # downloads the binaries on first use
    except Exception:  # pragma: no cover - network / platform dependent
        return None, None
    return shutil.which("ffmpeg"), shutil.which("ffprobe")


def find_binaries(ffmpeg: str | None = None, ffprobe: str | None = None) -> tuple[str, str]:
    """Locate ffmpeg and ffprobe: explicit paths, env vars, PATH, then static-ffmpeg."""
    ff = ffmpeg or os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    fp = ffprobe or os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
    if ff and fp and Path(ff).exists() and Path(fp).exists():
        return str(ff), str(fp)
    if ff and not fp and Path(ff).exists():
        sibling = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if sibling.exists():
            return str(ff), str(sibling)
    s_ff, s_fp = _which_static_ffmpeg()
    ff = ff if ff and Path(ff).exists() else s_ff
    fp = fp if fp and Path(fp).exists() else s_fp
    if not ff or not fp:
        raise MediaError(
            "ffmpeg/ffprobe not found. Install ffmpeg and put it on PATH, set FFMPEG_BIN / "
            "FFPROBE_BIN, or `pip install static-ffmpeg` in the orchestrator environment."
        )
    return str(ff), str(fp)


def _parse_rate(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _q(path: Path) -> str:
    """Quote a path for the ffmpeg concat demuxer (single quotes, forward slashes)."""
    return "'" + path.resolve().as_posix().replace("'", r"'\''") + "'"


class FFmpeg:
    def __init__(self, ffmpeg: str | None = None, ffprobe: str | None = None,
                 log: LogFn | None = None, timeout: int = 3600):
        self.ffmpeg, self.ffprobe = find_binaries(ffmpeg or None, ffprobe or None)
        self.log = log or (lambda _msg: None)
        self.timeout = timeout

    # Core ------------------------------------------------------------------
    def run(self, args: Sequence[str | os.PathLike], *, binary: str | None = None) -> str:
        cmd = [binary or self.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *map(str, args)]
        self.log("$ " + " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise MediaError(f"ffmpeg timed out after {self.timeout}s: {' '.join(cmd)}") from exc
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise MediaError(f"ffmpeg failed (exit {proc.returncode}):\n{' '.join(cmd)}\n{tail}")
        return proc.stdout

    def probe(self, path: str | os.PathLike) -> MediaInfo:
        path = Path(path)
        if not path.exists():
            raise MediaError(f"File not found: {path}")
        cmd = [self.ffprobe, "-v", "error", "-print_format", "json", "-show_format",
               "-show_streams", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=120, check=False)
        if proc.returncode != 0:
            raise MediaError(f"ffprobe failed for {path}: {proc.stderr.strip()[-500:]}")
        data = json.loads(proc.stdout or "{}")
        info = MediaInfo(duration=0.0, has_video=False, has_audio=False)
        try:
            info.duration = float(data.get("format", {}).get("duration", 0) or 0)
        except (TypeError, ValueError):
            info.duration = 0.0
        for stream in data.get("streams", []):
            kind = stream.get("codec_type")
            if kind == "video" and not info.has_video:
                info.has_video = True
                info.width = int(stream.get("width") or 0)
                info.height = int(stream.get("height") or 0)
                info.fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
                if not info.duration:
                    try:
                        info.duration = float(stream.get("duration") or 0)
                    except (TypeError, ValueError):
                        pass
            elif kind == "audio" and not info.has_audio:
                info.has_audio = True
                info.audio_sample_rate = int(stream.get("sample_rate") or 0)
                if not info.duration:
                    try:
                        info.duration = float(stream.get("duration") or 0)
                    except (TypeError, ValueError):
                        pass
        return info

    def duration(self, path: str | os.PathLike) -> float:
        return self.probe(path).duration

    # Pipeline operations ------------------------------------------------------
    def extract_reference_audio(self, src: Path, out: Path, seconds: float, *,
                                sample_rate: int = 24000, loudnorm: bool = True,
                                denoise: bool = False) -> Path:
        """Mono 16-bit WAV of the first ``seconds`` of speech (leading silence removed)."""
        filters = []
        if denoise:
            filters.append("afftdn=nf=-25")
        filters.append("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15")
        if loudnorm:
            filters.append("loudnorm=I=-18:TP=-1.5:LRA=11")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run(["-y", "-i", src, "-vn", "-af", ",".join(filters), "-ac", "1",
                  "-ar", str(sample_rate), "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", out])
        return out

    def prepare_driver_video(self, src: Path, out: Path, *, max_height: int, fps: int,
                             max_seconds: float, crf: int = 18) -> Path:
        """Silent, constant-fps, even-sized H.264 copy of the source, capped in height and length."""
        scale = (f"scale='trunc(iw*min(1,{max_height}/ih)/2)*2':"
                 f"'trunc(ih*min(1,{max_height}/ih)/2)*2'")
        vf = f"{scale},fps={fps},format=yuv420p"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run(["-y", "-i", src, "-an", "-t", f"{max_seconds:.3f}", "-vf", vf,
                  "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                  "-movflags", "+faststart", out])
        return out

    def extend_video_pingpong(self, src: Path, out: Path, target_seconds: float, *,
                              fps: int, crf: int = 18) -> Path:
        """Trim or ping-pong loop (forward, reverse, forward...) a silent video to ``target_seconds``."""
        duration = self.duration(src)
        if duration <= 0:
            raise MediaError(f"Could not determine duration of {src}")
        out.parent.mkdir(parents=True, exist_ok=True)
        encode = ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                  "-r", str(fps), "-movflags", "+faststart"]
        if duration >= target_seconds:
            self.run(["-y", "-i", src, "-an", "-t", f"{target_seconds:.3f}", *encode, out])
            return out
        reverse = out.with_name(out.stem + "_reverse.mp4")
        self.run(["-y", "-i", src, "-an", "-vf", "reverse", *encode, reverse])
        segments = math.ceil(target_seconds / duration)
        list_file = out.with_name(out.stem + "_concat.txt")
        lines = []
        for i in range(segments):
            lines.append(f"file {_q(src if i % 2 == 0 else reverse)}")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.run(["-y", "-f", "concat", "-safe", "0", "-i", list_file, "-an",
                  "-t", f"{target_seconds:.3f}", *encode, out])
        return out

    def mux(self, video: Path, audio: Path, out: Path, *, crf: int = 18, preset: str = "medium",
            audio_bitrate: str = "192k", copy_video: bool = False) -> Path:
        """Combine a video stream with an audio file; output ends with the shorter of the two."""
        out.parent.mkdir(parents=True, exist_ok=True)
        video_codec = ["-c:v", "copy"] if copy_video else [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        self.run(["-y", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
                  *video_codec, "-c:a", "aac", "-b:a", audio_bitrate, "-shortest",
                  "-movflags", "+faststart", out])
        return out

    def tone(self, out: Path, seconds: float, *, sample_rate: int = 24000, frequency: int = 220) -> Path:
        """Synthetic audio used by the mock TTS backend."""
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run(["-y", "-f", "lavfi", "-i",
                  f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={seconds:.3f}",
                  "-af", "tremolo=f=4:d=0.8", "-ac", "1", "-c:a", "pcm_s16le", out])
        return out
