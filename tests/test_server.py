import time

import pytest
from fastapi.testclient import TestClient

from avatar_pipeline.server import create_app


@pytest.fixture
def client(mock_cfg, ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")
    with TestClient(create_app(mock_cfg)) as c:
        yield c


def _wait(client, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish in time")


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] and "wav2lip" in body["backends"]["lipsync"] and "turbo" in body["tts_models"]


def test_validation(client, sample_video):
    with open(sample_video, "rb") as fh:
        r = client.post("/api/jobs", files={"video": ("clip.mp4", fh, "video/mp4")}, data={"script": "   "})
    assert r.status_code == 422
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/result").status_code == 404


def test_job_lifecycle(client, sample_video, sample_script):
    with open(sample_video, "rb") as fh:
        r = client.post("/api/jobs", files={"video": ("clip.mp4", fh, "video/mp4")},
                        data={"script": sample_script, "tts_backend": "mock", "lipsync_backend": "mock"})
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409

    job = _wait(client, job_id)
    assert job["status"] == "done", job
    assert job["progress"] == 100.0

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200 and result.headers["content-type"] == "video/mp4"
    assert len(result.content) > 10_000
    assert "== stage: finalize" in client.get(f"/api/jobs/{job_id}/log").text
    assert any(j["id"] == job_id for j in client.get("/api/jobs").json())
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
