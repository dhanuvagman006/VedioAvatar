from __future__ import annotations

from pathlib import Path

import pytest

from avatar_pipeline.config import load_config
from avatar_pipeline.media import MediaError, find_binaries
from avatar_pipeline.paths import REPO_ROOT

SAMPLE_VIDEO = REPO_ROOT / "examples" / "sample_face.mp4"
SAMPLE_SCRIPT = REPO_ROOT / "examples" / "sample_script.txt"


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    try:
        find_binaries()
        return True
    except MediaError:
        return False


@pytest.fixture
def sample_video(ffmpeg_available: bool) -> Path:
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")
    assert SAMPLE_VIDEO.exists()
    return SAMPLE_VIDEO


@pytest.fixture
def sample_script() -> str:
    return SAMPLE_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture
def mock_cfg(tmp_path: Path):
    """Config that keeps every artefact under tmp_path and uses the mock GPU backends."""
    cfg = load_config(REPO_ROOT / "config.yaml")
    return cfg.with_overrides({
        "paths": {"data_dir": str(tmp_path / "data")},
        "tts": {"backend": "mock"},
        "lipsync": {"backend": "mock"},
    })
