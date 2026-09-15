"""File-backed job store and a single background worker (the GPU serialises jobs anyway)."""
from __future__ import annotations

import json
import queue
import secrets
import shutil
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .pipeline import Pipeline, PipelineError

STATUSES = ("queued", "running", "done", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    input_video: str = ""
    ref_audio: str = ""
    input_name: str = ""
    script_chars: int = 0
    options: dict[str, Any] = field(default_factory=dict)
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def new_id(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)

    def job_dir(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
            raise ValueError("invalid job id")
        return self.root / job_id

    def create(self, **fields: Any) -> Job:
        with self._lock:
            job = Job(id=self.new_id(), **fields)
            self.job_dir(job.id).mkdir(parents=True, exist_ok=False)
            self.save(job)
            return job

    def save(self, job: Job) -> None:
        job.updated_at = _now()
        path = self.job_dir(job.id) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
            tmp.replace(path)

    def get(self, job_id: str) -> Job | None:
        try:
            path = self.job_dir(job_id) / "job.json"
        except ValueError:
            return None
        if not path.exists():
            return None
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in Job.__dataclass_fields__}
        return Job(**known)

    def list(self, limit: int = 50) -> list[Job]:
        jobs = [j for j in (self.get(p.name) for p in self.root.iterdir() if p.is_dir()) if j]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            d = self.job_dir(job_id)
            if not d.exists():
                return False
            shutil.rmtree(d, ignore_errors=True)
            return True

    def append_log(self, job_id: str, line: str) -> None:
        with open(self.job_dir(job_id) / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")

    def read_log(self, job_id: str, tail: int = 200) -> str:
        path = self.job_dir(job_id) / "log.txt"
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:] if tail else lines)


class JobRunner:
    """Runs queued jobs one at a time on a background thread."""

    def __init__(self, store: JobStore, cfg: Config):
        self.store = store
        self.cfg = cfg
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self.current: str | None = None

    def start(self) -> None:
        # Recover state from a previous server process.
        for job in self.store.list(limit=10_000):
            if job.status == "running":
                job.status, job.error, job.finished_at = "failed", "interrupted by a server restart", _now()
                self.store.save(job)
            elif job.status == "queued":
                self._queue.put(job.id)
        self._thread = threading.Thread(target=self._loop, name="avatar-job-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._run(job_id)
            except Exception:  # pragma: no cover - last-resort guard for the thread
                traceback.print_exc()
            finally:
                self.current = None

    def _run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job.status != "queued":
            return
        self.current = job_id
        job.status, job.started_at, job.message = "running", _now(), "starting"
        self.store.save(job)
        job_dir = self.store.job_dir(job_id)
        last_saved = [-1.0]

        def log(line: str) -> None:
            self.store.append_log(job_id, line)

        def progress(pct: float, stage: str, message: str) -> None:
            job.progress, job.stage = round(pct, 1), stage
            if message:
                job.message = message
            if abs(pct - last_saved[0]) >= 0.5 or message:
                last_saved[0] = pct
                self.store.save(job)

        try:
            script = (job_dir / "script.txt").read_text(encoding="utf-8")
            result = Pipeline(self.cfg, log=log, progress=progress).run(
                video=job_dir / job.input_video,
                script_text=script,
                out_path=job_dir / "output.mp4",
                workdir=job_dir / "work",
                options=job.options,
                ref_audio=(job_dir / job.ref_audio) if job.ref_audio else None,
            )
            job.status, job.progress, job.message = "done", 100.0, "finished"
            job.result = result.output.name
        except PipelineError as exc:
            job.status, job.error, job.message = "failed", str(exc), "failed"
            log("ERROR: " + str(exc))
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.message = "failed", f"{type(exc).__name__}: {exc}", "failed"
            log("ERROR: " + traceback.format_exc())
        finally:
            job.finished_at = _now()
            self.store.save(job)
