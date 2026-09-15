import json
from pathlib import Path

import pytest

from avatar_pipeline.media import FFmpeg
from avatar_pipeline.pipeline import Pipeline, PipelineError, options_to_overrides


def test_options_to_overrides_maps_flat_keys():
    assert options_to_overrides({"tts_model": "nano", "lipsync_backend": "wav2lip", "language": None}) == {
        "tts": {"model": "nano"}, "lipsync": {"backend": "wav2lip"}}
    assert options_to_overrides({"unknown": 1}) == {}
    assert options_to_overrides({"tts": {"device": "cpu"}}) == {"tts": {"device": "cpu"}}


def test_end_to_end_with_mock_backends(mock_cfg, sample_video, sample_script, tmp_path: Path):
    events = []
    logs = []
    pipeline = Pipeline(mock_cfg, log=logs.append, progress=lambda p, s, m: events.append((p, s)))
    result = pipeline.run(sample_video, sample_script, tmp_path / "out.mp4", tmp_path / "work")

    assert result.output.exists()
    ff = FFmpeg()
    speech = ff.duration(tmp_path / "work" / "speech.wav")
    out = ff.probe(result.output)
    assert out.has_video and out.has_audio
    assert speech > 20  # the mock estimates ~34 s for the sample script
    assert abs(out.duration - speech) < 0.3
    assert out.height == 720

    manifest = json.loads((tmp_path / "work" / "manifest.json").read_text())
    assert manifest["tts_backend"] == "mock" and manifest["lipsync_backend"] == "mock"
    assert set(manifest["stages"]) == {"probe", "reference", "driver", "tts", "extend", "lipsync", "finalize"}
    assert events[-1][0] == 100.0 and events[-1][1] == "finalize"
    assert any("== stage: lipsync" in line for line in logs)


def test_empty_script_fails_fast(mock_cfg, sample_video, tmp_path: Path):
    with pytest.raises(PipelineError, match="empty"):
        Pipeline(mock_cfg).run(sample_video, "   ", tmp_path / "out.mp4", tmp_path / "work")


def test_missing_backend_environment_is_reported(mock_cfg, sample_video, sample_script, tmp_path: Path):
    cfg = mock_cfg.with_overrides({"paths": {"envs_dir": str(tmp_path / "no-envs")}})
    with pytest.raises(PipelineError, match="setup_envs"):
        Pipeline(cfg).run(sample_video, sample_script, tmp_path / "out.mp4", tmp_path / "work",
                          options={"tts_backend": "chatterbox"})
