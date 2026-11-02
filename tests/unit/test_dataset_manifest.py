"""Tests use dynamically generated synthetic audio/protocol fixtures only.

These temporary files are NOT real ASVspoof or WaveFake dataset records and
must never be cited as dataset examples, model results, or benchmark metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.dataset_manifest import (
    ASVSPOOF_SOURCE,
    WAVEFAKE_SOURCE,
    ManifestError,
    ManifestRecord,
    build_asvspoof_records,
    build_wavefake_records,
    create_quality_report,
    parse_protocol_file,
    write_manifest,
)


def _write_synthetic_wav(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Low-amplitude synthetic silence-like waveform, generated only in tmp_path.
    sf.write(path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)


def _write_protocol(protocol_dir: Path, split: str, line: str) -> None:
    protocol_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"train": "trn", "dev": "trl", "eval": "trl"}[split]
    filename = f"ASVspoof2019.LA.cm.{split}.{suffix}.txt"
    (protocol_dir / filename).write_text(line + "\n", encoding="utf-8")


def test_asvspoof_protocols_keep_official_splits_and_map_labels(tmp_path: Path) -> None:
    """Synthetic protocol/audio fixtures test parser behavior, not benchmark data."""
    protocol_dir = tmp_path / "protocols"
    audio_root = tmp_path / "audio"
    fixture_rows = {
        "train": "speaker-train LA_T_SYNTH_REAL - - bonafide",
        "dev": "speaker-dev LA_D_SYNTH_A01 - A07 spoof",
        "eval": "speaker-eval LA_E_SYNTH_A02 - A19 spoof",
    }
    for split, line in fixture_rows.items():
        _write_protocol(protocol_dir, split, line)
        utterance_id = line.split()[1]
        suffix = "flac"
        _write_synthetic_wav(audio_root / f"ASVspoof2019_LA_{split}" / f"{utterance_id}.{suffix}")

    records = build_asvspoof_records(protocol_dir, audio_root)
    assert [(record.split, record.label) for record in records] == [
        ("train", 0),
        ("dev", 1),
        ("eval", 1),
    ]
    assert [record.source for record in records] == [ASVSPOOF_SOURCE] * 3
    assert [record.attack_type for record in records] == ["", "A07", "A19"]
    assert all(Path(record.file_path).is_file() for record in records)

    manifest_path = tmp_path / "manifests" / "sample.csv"
    assert write_manifest(records, manifest_path) == 3
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert [row["label"] for row in rows] == ["0", "1", "1"]
    assert [row["split"] for row in rows] == ["train", "dev", "eval"]

    report = create_quality_report(records)
    assert report["audio_validation"]["missing_paths"] == 0
    assert report["audio_validation"]["unreadable_paths"] == 0
    assert report["audio_validation"]["sample_rate_mismatches"] == 0
    assert report["leakage_checks"]["cross_split_speaker_groups"] == 0
    assert report["class_distribution_by_split"]["train"]["bona_fide_label_0"] == 1


def test_missing_audio_and_cross_split_speaker_are_reported(tmp_path: Path) -> None:
    """Report a missing synthetic fixture and speaker overlap without hiding it."""
    protocol_dir = tmp_path / "protocols"
    audio_root = tmp_path / "audio"
    _write_protocol(protocol_dir, "train", "shared-speaker LA_T_PRESENT - - bonafide")
    _write_protocol(protocol_dir, "dev", "shared-speaker LA_D_MISSING - A02 spoof")
    _write_protocol(protocol_dir, "eval", "eval-speaker LA_E_PRESENT - A03 spoof")
    _write_synthetic_wav(audio_root / "train" / "LA_T_PRESENT.flac")
    _write_synthetic_wav(audio_root / "eval" / "LA_E_PRESENT.flac")

    report = create_quality_report(build_asvspoof_records(protocol_dir, audio_root))
    assert report["audio_validation"]["missing_paths"] == 1
    assert report["audio_validation"]["empty_path_records"] == 1
    assert report["leakage_checks"]["cross_split_speaker_groups"] == 1
    assert report["issue_count"] >= 2


def test_unrecognized_protocol_key_is_rejected(tmp_path: Path) -> None:
    protocol = tmp_path / "unknown-key.txt"
    protocol.write_text("speaker utt - - unknown\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="unsupported/missing protocol key"):
        parse_protocol_file(protocol, "train")


def test_wavefake_records_are_external_spoof_only(tmp_path: Path) -> None:
    """Synthetic directory fixture checks external-test labeling, not WaveFake contents."""
    archive_root = tmp_path / "generated_audio"
    _write_synthetic_wav(archive_root / "vocoder-fixture" / "sample-001.wav")
    records = build_wavefake_records(archive_root)
    assert len(records) == 1
    assert records[0].label == 1
    assert records[0].split == "external_eval"
    assert records[0].source == WAVEFAKE_SOURCE
    assert records[0].attack_type == "vocoder-fixture"


def test_audio_validation_reports_rate_mismatch_and_unreadable_file(tmp_path: Path) -> None:
    """Malformed and off-rate fixture files are reported without crashing."""
    off_rate = tmp_path / "off-rate.wav"
    _write_synthetic_wav(off_rate, sample_rate=22050)
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_text("not audio", encoding="utf-8")
    records = [
        ManifestRecord(str(off_rate), 0, "train", ASVSPOOF_SOURCE, utterance_id="SYNTH_RATE"),
        ManifestRecord(str(corrupt), 1, "dev", ASVSPOOF_SOURCE, utterance_id="SYNTH_CORRUPT"),
    ]

    report = create_quality_report(records, expected_sample_rate=16000)
    assert report["audio_validation"]["sample_rate_mismatches"] == 1
    assert report["audio_validation"]["unreadable_paths"] == 1
