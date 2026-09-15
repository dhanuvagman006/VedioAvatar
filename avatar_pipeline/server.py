"""FastAPI service: upload a clip + script, poll progress, download the result."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from . import __version__
from .backends import list_backends
from .backends.tts_chatterbox import MODELS as TTS_MODELS
from .config import Config, load_config
from .doctor import quick_checks
from .gpu import gpu_info
from .jobs import JobRunner, JobStore
from .paths import WEB_DIR

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4"}


async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Upload exceeds {max_bytes // (1 << 20)} MB")
            fh.write(chunk)
    return size


def _suffix(upload: UploadFile, allowed: set[str], default: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    return suffix if suffix in allowed else default


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    store = JobStore(cfg.data_dir / "jobs")
    runner = JobRunner(store, cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runner.start()
        yield
        runner.stop()

    app = FastAPI(title="VedioAvatar", version=__version__, lifespan=lifespan)
    app.state.cfg, app.state.store, app.state.runner = cfg, store, runner
    max_bytes = int(cfg.get("server.max_upload_mb", 500)) * (1 << 20)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "gpu": gpu_info(),
            "backends": list_backends(),
            "tts_models": list(TTS_MODELS),
            "defaults": {"tts": cfg.section("tts"), "lipsync": {
                "backend": cfg.get("lipsync.backend"),
                "auto_latentsync_min_vram_mb": cfg.get("lipsync.auto_latentsync_min_vram_mb")}},
            "checks": [c.__dict__ for c in quick_checks(cfg)],
            "queue": runner.queue_size(),
            "running": runner.current,
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        request: Request,
        video: UploadFile = File(...),
        script: str | None = Form(None),
        script_file: UploadFile | None = File(None),
        ref_audio: UploadFile | None = File(None),
        tts_backend: str | None = Form(None),
        tts_model: str | None = Form(None),
        language: str | None = Form(None),
        lipsync_backend: str | None = Form(None),
        exaggeration: float | None = Form(None),
        cfg_weight: float | None = Form(None),
        temperature: float | None = Form(None),
        denoise: bool | None = Form(None),
    ) -> dict[str, Any]:
        if script_file is not None and script_file.filename:
            raw = await script_file.read()
            if len(raw) > 200_000:
                raise HTTPException(status_code=413, detail="Script file is too large (200 KB max)")
            script_text = raw.decode("utf-8", errors="replace")
        else:
            script_text = script or ""
        if not script_text.strip():
            raise HTTPException(status_code=422, detail="A non-empty script is required")
        if tts_model and tts_model not in TTS_MODELS:
            raise HTTPException(status_code=422, detail=f"tts_model must be one of {', '.join(TTS_MODELS)}")
        backends = list_backends()
        if tts_backend and tts_backend not in backends["tts"]:
            raise HTTPException(status_code=422, detail=f"tts_backend must be one of {', '.join(backends['tts'])}")
        if lipsync_backend and lipsync_backend not in ("auto", *backends["lipsync"]):
            raise HTTPException(status_code=422, detail=f"lipsync_backend must be auto or one of {', '.join(backends['lipsync'])}")

        options = {k: v for k, v in {
            "tts_backend": tts_backend, "tts_model": tts_model, "language": language,
            "lipsync_backend": lipsync_backend, "exaggeration": exaggeration,
            "cfg_weight": cfg_weight, "temperature": temperature, "denoise": denoise,
        }.items() if v not in (None, "")}

        video_name = "input_video" + _suffix(video, VIDEO_SUFFIXES, ".mp4")
        job = store.create(input_video=video_name, input_name=video.filename or "",
                           script_chars=len(script_text), options=options)
        job_dir = store.job_dir(job.id)
        try:
            await _save_upload(video, job_dir / video_name, max_bytes)
            if ref_audio is not None and ref_audio.filename:
                job.ref_audio = "ref_audio" + _suffix(ref_audio, AUDIO_SUFFIXES, ".wav")
                await _save_upload(ref_audio, job_dir / job.ref_audio, max_bytes)
            (job_dir / "script.txt").write_text(script_text, encoding="utf-8")
        except HTTPException:
            store.delete(job.id)
            raise
        store.save(job)
        runner.submit(job.id)
        return job.to_dict()

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
        return [j.to_dict() for j in store.list(limit=limit)]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    async def get_log(job_id: str, tail: int = 200) -> str:
        if store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return store.read_log(job_id, tail=tail)

    @app.get("/api/jobs/{job_id}/result")
    async def get_result(job_id: str) -> FileResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "done" or not job.result:
            raise HTTPException(status_code=409, detail=f"job is {job.status}")
        path = store.job_dir(job_id) / job.result
        return FileResponse(path, media_type="video/mp4", filename=f"avatar_{job_id}.mp4")

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> JSONResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status == "running":
            raise HTTPException(status_code=409, detail="job is running")
        store.delete(job_id)
        return JSONResponse({"deleted": job_id})

    return app


app = create_app()
