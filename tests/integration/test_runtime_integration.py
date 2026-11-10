"""Runtime integration checks; FFmpeg test uses the real local executable."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import process_podcast_dataset as processing

MANIFEST_FIELDS = (
    "source_id",
    "youtube_url",
    "language",
    "permission_status",
    "original_video",
    "mp3_file",
    "wav_folder",
    "processing_status",
)


def _write_pipeline_manifest(root: Path) -> Path:
    dataset = root / "PodcastDataset"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset / "processing_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "INT001",
                "youtube_url": "https://example.invalid/not-a-media-input",
                "language": "unknown",
                "permission_status": "granted",
                "original_video": "PodcastDataset/originals/INT001",
                "mp3_file": "PodcastDataset/mp3_audio/INT001.mp3",
                "wav_folder": "PodcastDataset/clips/INT001",
                "processing_status": "waiting_for_audio",
            }
        )
    (dataset / "clip_metadata.csv").write_text(
        "clip_id,source_id,file_path,duration,sample_rate,language,speaker_id,quality_status\n",
        encoding="utf-8",
    )
    return manifest_path


def test_real_ffmpeg_converts_temporary_wav_without_touching_source(tmp_path: Path) -> None:
    try:
        ffmpeg = processing.resolve_ffmpeg()
    except processing.PipelineError as error:
        pytest.skip(str(error))

    source = tmp_path / "synthetic_fixture.wav"
    output = tmp_path / "synthetic_fixture.mp3"
    times = np.arange(200000, dtype=np.float32) / 16000
    sf.write(source, 0.1 * np.sin(2 * np.pi * 440 * times), 16000, subtype="PCM_16")
    source_before = source.read_bytes()

    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-n",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file() and output.stat().st_size > 0
    assert source.read_bytes() == source_before
    decoded = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(output), "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert decoded.returncode == 0, decoded.stderr

    clips = processing._segment_recording(ffmpeg, output, tmp_path / "clips")
    assert len(clips) == 2
    first_info = sf.info(clips[0])
    final_info = sf.info(clips[1])
    assert first_info.samplerate == processing.TARGET_SAMPLE_RATE
    assert first_info.channels == 1
    assert first_info.subtype == "PCM_16"
    assert first_info.duration == pytest.approx(processing.SEGMENT_SECONDS, abs=0.1)
    assert final_info.duration == pytest.approx(2.5, abs=0.1)


def test_ffmpeg_explicit_path_takes_precedence_and_is_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.touch()
    probes: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        probes.append(command)
        return SimpleNamespace(returncode=0, stdout="ffmpeg version fixture", stderr="")

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    monkeypatch.setattr(processing.shutil, "which", lambda _name: pytest.fail("must use explicit path"))

    resolved = processing.resolve_ffmpeg(executable)

    assert Path(resolved) == executable.resolve()
    assert probes == [[str(executable.resolve()), "-version"]]


def test_ffmpeg_environment_setting_and_path_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.touch()
    monkeypatch.setattr(
        processing.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="ffmpeg version fixture", stderr=""),
    )
    assert processing.resolve_ffmpeg(environ={"FFMPEG_PATH": str(executable)}) == str(executable.resolve())

    monkeypatch.setattr(processing.shutil, "which", lambda name: "C:/tools/ffmpeg.exe" if name == "ffmpeg" else None)
    assert processing.resolve_ffmpeg(environ={}) == "C:/tools/ffmpeg.exe"


def test_unavailable_or_invalid_ffmpeg_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(processing.shutil, "which", lambda _name: None)

    with pytest.raises(processing.PipelineError, match="FFmpeg is unavailable"):
        processing.resolve_ffmpeg(environ={})
    with pytest.raises(processing.PipelineError, match="configured by --ffmpeg"):
        processing.resolve_ffmpeg(tmp_path / "missing.exe", environ={})


def test_pyannote_receives_environment_token_without_exposing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyannote.audio import Pipeline

    token = "fixture-secret-token-do-not-log"
    captured: dict[str, object] = {}
    fake_pipeline = SimpleNamespace(to=lambda device: captured.update(device=device))

    def fake_from_pretrained(model_id: str, *, token: str | None) -> SimpleNamespace:
        captured.update(model_id=model_id, token=token)
        return fake_pipeline

    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(Pipeline, "from_pretrained", fake_from_pretrained)

    diarizer = processing.load_pyannote_diarizer("fixture/diarization-model")

    assert callable(diarizer)
    assert captured["model_id"] == "fixture/diarization-model"
    assert captured["token"] == token
    assert token not in repr(diarizer)


def test_pyannote_loader_failure_is_sanitized_and_cached_access_needs_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyannote.audio import Pipeline

    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fail_loader(_model_id: str, *, token: str | None) -> None:
        assert token is None
        raise RuntimeError("credential fixture-secret-token-do-not-log rejected")

    monkeypatch.setattr(Pipeline, "from_pretrained", fail_loader)
    with pytest.raises(processing.PipelineError) as error:
        processing.load_pyannote_diarizer("fixture/missing-cache")
    assert "fixture-secret-token-do-not-log" not in str(error.value)
    assert "HF_TOKEN" in str(error.value)


def test_cached_pyannote_diarizes_temporary_audio_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--pyannote--speaker-diarization-3.1"
    )
    refs_dir = cache / "refs"
    snapshots_dir = cache / "snapshots"
    if not refs_dir.is_dir() or not snapshots_dir.is_dir():
        pytest.skip("Pyannote diarization model is not cached; no download attempted")
    refs = list(refs_dir.iterdir())
    snapshot = snapshots_dir / refs[0].read_text(encoding="utf-8").strip() if refs else None
    config = snapshot / "config.yaml" if snapshot else None
    if config is None or not config.is_file():
        pytest.skip("Cached Pyannote config is incomplete; no download attempted")
    for model_dir in (
        "models--pyannote--segmentation-3.0",
        "models--pyannote--wespeaker-voxceleb-resnet34-LM",
    ):
        model_cache = Path.home() / ".cache" / "huggingface" / "hub" / model_dir
        if not model_cache.is_dir() or not any(model_cache.rglob("pytorch_model.bin")):
            pytest.skip(f"Required cached Pyannote dependency is missing: {model_dir}")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio_path = tmp_path / "synthetic_diarization_fixture.wav"
    sample_rate = 16000
    times = np.arange(sample_rate * 12, dtype=np.float32) / sample_rate
    waveform = (
        0.08
        * np.sin(2 * np.pi * (130 + 25 * np.sin(2 * np.pi * 0.3 * times)) * times)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * times))
    ).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")

    diarizer = processing.load_pyannote_diarizer(str(config))
    turns = diarizer(audio_path)

    assert isinstance(turns, list)
    assert all(turn.start >= 0 and turn.end > turn.start and turn.speaker_id for turn in turns)


def test_complete_local_pipeline_with_real_ffmpeg_and_cached_pyannote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        processing.resolve_ffmpeg()
    except processing.PipelineError as error:
        pytest.skip(str(error))

    diarization_cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--pyannote--speaker-diarization-3.1"
    )
    required_caches = (
        diarization_cache,
        Path.home() / ".cache" / "huggingface" / "hub" / "models--pyannote--segmentation-3.0",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--pyannote--wespeaker-voxceleb-resnet34-LM",
    )
    if any(not path.is_dir() for path in required_caches):
        pytest.skip("Cached Pyannote diarization dependencies are unavailable; no download attempted")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    manifest_path = _write_pipeline_manifest(tmp_path)
    input_audio = tmp_path / "PodcastDataset" / "originals" / "INT001" / "fixture.wav"
    input_audio.parent.mkdir(parents=True)
    sample_rate = 16000
    times = np.arange(sample_rate * 12, dtype=np.float32) / sample_rate
    waveform = (
        0.08
        * np.sin(2 * np.pi * (130 + 25 * np.sin(2 * np.pi * 0.3 * times)) * times)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * times))
    ).astype(np.float32)
    sf.write(input_audio, waveform, sample_rate, subtype="PCM_16")
    original_bytes = input_audio.read_bytes()

    result = processing.run_pipeline(
        tmp_path,
        {"INT001": input_audio.relative_to(tmp_path).as_posix()},
    )

    assert result.processed_count == 1
    assert result.error_count == 0
    assert input_audio.read_bytes() == original_bytes
    assert (tmp_path / "PodcastDataset" / "mp3_audio" / "INT001.mp3").is_file()
    clips = sorted((tmp_path / "PodcastDataset" / "clips" / "INT001").glob("*.wav"))
    assert len(clips) == 2
    assert all(sf.info(clip).samplerate == 16000 and sf.info(clip).channels == 1 for clip in clips)

    turns_path = tmp_path / "PodcastDataset" / "diarization" / "INT001_speakers.csv"
    assert turns_path.is_file()
    with turns_path.open(newline="", encoding="utf-8") as file:
        turns = list(csv.DictReader(file))
    assert tuple(turns[0]) == processing.SPEAKER_TURN_FIELDS

    metadata_path = tmp_path / "PodcastDataset" / "clip_metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as file:
        metadata = list(csv.DictReader(file))
    assert len(metadata) == 2
    assert all(row["source_id"] == "INT001" and row["sample_rate"] == "16000" for row in metadata)
    assert all(row["quality_status"] in {"PASS", "CHECK"} for row in metadata)

    with manifest_path.open(newline="", encoding="utf-8") as file:
        updated_manifest = list(csv.DictReader(file))
    assert updated_manifest[0]["processing_status"] == "processed"
    assert processing.run_pipeline(tmp_path).skipped_count == 1