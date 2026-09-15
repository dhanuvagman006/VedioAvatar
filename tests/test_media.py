from pathlib import Path

import pytest

from avatar_pipeline.media import FFmpeg


@pytest.fixture
def ff(ffmpeg_available):
    if not ffmpeg_available:
        pytest.skip("ffmpeg not available")
    return FFmpeg()


def test_probe_sample(ff, sample_video):
    info = ff.probe(sample_video)
    assert info.has_video and info.has_audio
    assert 19 < info.duration < 21
    assert info.width == 1080 and info.height == 1920
    assert round(info.fps) == 25


def test_reference_audio_is_trimmed_mono(ff, sample_video, tmp_path: Path):
    out = ff.extract_reference_audio(sample_video, tmp_path / "ref.wav", 6.0)
    info = ff.probe(out)
    assert info.has_audio and not info.has_video
    assert 5.5 <= info.duration <= 6.05
    assert info.audio_sample_rate == 24000


def test_driver_video_is_capped_and_silent(ff, sample_video, tmp_path: Path):
    out = ff.prepare_driver_video(sample_video, tmp_path / "driver.mp4", max_height=720, fps=25, max_seconds=8)
    info = ff.probe(out)
    assert not info.has_audio
    assert info.height == 720 and info.width == 404  # even width, aspect kept
    assert round(info.fps) == 25
    assert 7.9 <= info.duration <= 8.1


def test_pingpong_extends_to_target(ff, sample_video, tmp_path: Path):
    driver = ff.prepare_driver_video(sample_video, tmp_path / "driver.mp4", max_height=240, fps=25, max_seconds=4)
    out = ff.extend_video_pingpong(driver, tmp_path / "long.mp4", 9.5, fps=25)
    assert abs(ff.duration(out) - 9.5) < 0.15
    assert (tmp_path / "long_reverse.mp4").exists()


def test_pingpong_trims_when_source_is_longer(ff, sample_video, tmp_path: Path):
    driver = ff.prepare_driver_video(sample_video, tmp_path / "driver.mp4", max_height=240, fps=25, max_seconds=6)
    out = ff.extend_video_pingpong(driver, tmp_path / "short.mp4", 2.0, fps=25)
    assert abs(ff.duration(out) - 2.0) < 0.15


def test_tone_and_mux(ff, sample_video, tmp_path: Path):
    driver = ff.prepare_driver_video(sample_video, tmp_path / "driver.mp4", max_height=240, fps=25, max_seconds=5)
    tone = ff.tone(tmp_path / "tone.wav", 3.0)
    out = ff.mux(driver, tone, tmp_path / "out.mp4")
    info = ff.probe(out)
    assert info.has_video and info.has_audio
    assert abs(info.duration - 3.0) < 0.2
