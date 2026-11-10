from __future__ import annotations

import csv
import json
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
LEGACY_METADATA_HEADER = "clip_id,source_id,file_path,duration,sample_rate,language,speaker_id,quality_status\n"


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    dataset = root / "PodcastDataset"
    dataset.mkdir(parents=True, exist_ok=True)
    path = dataset / "processing_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _manifest_row(source_id: str = "P001", **updates: str) -> dict[str, str]:
    row = {
        "source_id": source_id,
        "youtube_url": "https://www.youtube.com/watch?v=fixture-only",
        "language": "unknown",
        "permission_status": "granted",
        "original_video": f"PodcastDataset/originals/{source_id}",
        "mp3_file": f"PodcastDataset/mp3_audio/{source_id}.mp3",
        "wav_folder": f"PodcastDataset/clips/{source_id}",
        "processing_status": "waiting_for_audio",
    }
    row.update(updates)
    return row


def _write_synthetic_audio(path: Path, duration: float = 12.5) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(16000 * duration)
    waveform = (0.1 * np.sin(2 * np.pi * 220 * np.arange(samples) / 16000)).astype(np.float32)
    sf.write(path, waveform, 16000, subtype="PCM_16")
    return path.read_bytes()


def _install_fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    converted_sources: dict[Path, Path] = {}

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[-1])
        if "segment" not in command:
            source_path = input_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"synthetic test MP3 placeholder")
            converted_sources[output_path] = source_path
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        original = converted_sources[input_path]
        samples, sample_rate = sf.read(original, dtype="float32")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples_per_clip = 10 * sample_rate
        for index, start in enumerate(range(0, len(samples), samples_per_clip)):
            segment = samples[start : start + samples_per_clip]
            clip_path = Path(str(output_path).replace("%04d", f"{index:04d}"))
            sf.write(clip_path, segment, sample_rate, subtype="PCM_16")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    monkeypatch.setattr(processing, "resolve_ffmpeg", lambda _configured=None: "fake-ffmpeg.exe")
    return calls


def test_pipeline_waits_for_local_media_without_touching_manifest_or_metadata(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_manifest_row()])
    metadata_path = tmp_path / "PodcastDataset" / "clip_metadata.csv"
    metadata_path.write_text(LEGACY_METADATA_HEADER, encoding="utf-8")
    manifest_before = manifest_path.read_bytes()
    metadata_before = metadata_path.read_bytes()

    result = processing.run_pipeline(tmp_path)

    assert result.waiting_count == 1
    assert result.error_count == 0
    assert "waiting for an explicitly supplied local media file" in result.format_text()
    assert manifest_path.read_bytes() == manifest_before
    assert metadata_path.read_bytes() == metadata_before


def test_pipeline_processes_complete_recording_before_segmentation_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_id = "P001"
    input_audio = tmp_path / "PodcastDataset" / "originals" / source_id / "fixture.wav"
    original_bytes = _write_synthetic_audio(input_audio)
    manifest_path = _write_manifest(tmp_path, [_manifest_row()])
    metadata_path = tmp_path / "PodcastDataset" / "clip_metadata.csv"
    metadata_path.write_text(LEGACY_METADATA_HEADER, encoding="utf-8")
    ffmpeg_calls = _install_fake_ffmpeg(monkeypatch)
    diarized_paths: list[Path] = []

    def fake_diarizer(path: Path) -> list[processing.SpeakerTurn]:
        diarized_paths.append(path)
        assert path.is_file()
        return [processing.SpeakerTurn(0.0, 4.0, "speaker_0"), processing.SpeakerTurn(9.0, 12.5, "speaker_1")]

    result = processing.run_pipeline(
        tmp_path,
        {source_id: input_audio.relative_to(tmp_path).as_posix()},
        diarizer=fake_diarizer,
    )

    assert result.processed_count == 1
    assert result.error_count == 0
    assert len(diarized_paths) == 1
    assert len(ffmpeg_calls) == 2
    assert "segment" not in ffmpeg_calls[0]
    assert "segment" in ffmpeg_calls[1]
    assert input_audio.read_bytes() == original_bytes

    clips = sorted((tmp_path / "PodcastDataset" / "clips" / source_id).glob("*.wav"))
    assert len(clips) == 2
    with metadata_path.open(newline="", encoding="utf-8") as file:
        metadata = list(csv.DictReader(file))
    assert [row["clip_id"] for row in metadata] == ["P001_0000", "P001_0001"]
    assert metadata[0]["speaker_ids"] == "speaker_0;speaker_1"
    assert metadata[1]["speaker_ids"] == "speaker_1"
    assert metadata[0]["sample_rate"] == "16000"
    assert metadata[0]["quality_status"] == "PASS"
    assert float(metadata[1]["clip_end_seconds"]) == pytest.approx(12.5)
    first_clip_turns = json.loads(metadata[0]["speaker_turns"])
    second_clip_turns = json.loads(metadata[1]["speaker_turns"])
    assert first_clip_turns[-1]["start_seconds"] == 9.0
    assert first_clip_turns[-1]["end_seconds"] == 10.0
    assert second_clip_turns[0]["start_seconds"] == 10.0
    assert second_clip_turns[0]["end_seconds"] == 12.5

    with manifest_path.open(newline="", encoding="utf-8") as file:
        updated_manifest = list(csv.DictReader(file))
    assert updated_manifest[0]["processing_status"] == "processed"
    assert updated_manifest[0]["original_video"] == input_audio.relative_to(tmp_path).as_posix()
    speaker_csv = tmp_path / "PodcastDataset" / "diarization" / "P001_speakers.csv"
    assert speaker_csv.is_file()
    with speaker_csv.open(newline="", encoding="utf-8") as file:
        turns = list(csv.DictReader(file))
    assert tuple(turns[0]) == processing.SPEAKER_TURN_FIELDS
    assert [(row["speaker_id"], row["start_seconds"], row["end_seconds"]) for row in turns] == [
        ("speaker_0", "0.000", "4.000"),
        ("speaker_1", "9.000", "12.500"),
    ]

    rerun = processing.run_pipeline(tmp_path, diarizer=lambda _path: pytest.fail("must not diarize on rerun"))
    assert rerun.skipped_count == 1
    assert len(ffmpeg_calls) == 2


@pytest.mark.parametrize(
    ("permission", "filename", "expected"),
    [
        ("pending", "recording.wav", "permission_status"),
        ("granted", "recording.xyz", "Unsupported media extension"),
        ("granted", "https://youtube.com/watch?v=not-a-file", "URL, not a local file"),
    ],
)
def test_invalid_media_inputs_are_rejected_without_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    permission: str,
    filename: str,
    expected: str,
) -> None:
    row = _manifest_row(permission_status=permission)
    _write_manifest(tmp_path, [row])
    media = tmp_path / "PodcastDataset" / "originals" / "P001" / "recording.wav"
    _write_synthetic_audio(media, duration=1)
    calls = _install_fake_ffmpeg(monkeypatch)

    result = processing.run_pipeline(tmp_path, {"P001": filename})

    assert result.error_count == 1
    assert expected in result.sources[0].message
    assert not calls
    assert row["processing_status"] == "waiting_for_audio"


def test_existing_output_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_audio = tmp_path / "PodcastDataset" / "originals" / "P001" / "input.wav"
    _write_synthetic_audio(input_audio, duration=1)
    manifest_path = _write_manifest(tmp_path, [_manifest_row()])
    output = tmp_path / "PodcastDataset" / "mp3_audio" / "P001.mp3"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep existing content")
    calls = _install_fake_ffmpeg(monkeypatch)

    result = processing.run_pipeline(
        tmp_path,
        {"P001": input_audio.relative_to(tmp_path).as_posix()},
        diarizer=lambda _path: [],
    )

    assert result.error_count == 1
    assert not calls
    assert output.read_bytes() == b"keep existing content"
    assert next(csv.DictReader(manifest_path.open(encoding="utf-8")))['processing_status'] == "waiting_for_audio"


def test_failed_ffmpeg_does_not_update_manifest_or_publish_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_audio = tmp_path / "PodcastDataset" / "originals" / "P001" / "input.wav"
    original_bytes = _write_synthetic_audio(input_audio, duration=1)
    manifest_path = _write_manifest(tmp_path, [_manifest_row()])
    manifest_before = manifest_path.read_bytes()
    metadata_path = tmp_path / "PodcastDataset" / "clip_metadata.csv"
    metadata_path.write_text(LEGACY_METADATA_HEADER, encoding="utf-8")
    metadata_before = metadata_path.read_bytes()
    calls = _install_fake_ffmpeg(monkeypatch)

    def fail_conversion(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(processing.subprocess, "run", fail_conversion)
    result = processing.run_pipeline(
        tmp_path,
        {"P001": input_audio.relative_to(tmp_path).as_posix()},
        diarizer=lambda _path: [],
    )

    assert result.error_count == 1
    assert manifest_path.read_bytes() == manifest_before
    assert metadata_path.read_bytes() == metadata_before
    assert input_audio.read_bytes() == original_bytes
    assert not (tmp_path / "PodcastDataset" / "mp3_audio" / "P001.mp3").exists()
    assert not (tmp_path / "PodcastDataset" / "clips" / "P001").exists()


def test_missing_explicit_file_reports_waiting_without_loading_pyannote(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [_manifest_row()])

    result = processing.run_pipeline(tmp_path, {"P001": "PodcastDataset/originals/P001/missing.wav"})

    assert result.waiting_count == 1
    assert result.error_count == 0


def test_output_path_must_be_unique_between_sources(tmp_path: Path) -> None:
    first = _manifest_row("P001")
    second = _manifest_row("P002", mp3_file=first["mp3_file"])
    _write_manifest(tmp_path, [first, second])

    with pytest.raises(processing.PipelineError, match="Duplicate output path"):
        processing.run_pipeline(tmp_path)