"""Synthetic temporary fixtures exercise readiness logic, not training data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.dataset import read_manifest_records
from training.dataset_readiness import (
    METADATA_FIELDS,
    audit_labeled_metadata,
    prepare_training_splits,
    validate_split_leakage,
    write_split_manifest,
)


def _metadata_row(index: int, label: int, *, speaker_id: str | None = None, source_id: str | None = None) -> dict[str, str]:
    is_spoof = label == 1
    return {
        "file_path": f"audio/clip_{index}.wav",
        "label": str(label),
        "split": "",
        "source_id": source_id or f"recording_{index}",
        "source": "temporary-test-fixture",
        "speaker_id": speaker_id or f"speaker_{index}",
        "utterance_id": f"utterance_{index}",
        "language": "fixture-language",
        "permission_status": "granted",
        "permission_evidence_ref": f"evidence/permission-{index}.md",
        "generation_method": "fixture_tts" if is_spoof else "not_applicable",
        "label_source": f"evidence/label-source-{index}.md",
        "label_provenance": f"evidence/label-review-{index}.md",
        "attack_type": "fixture_tts" if is_spoof else "",
    }


def _write_audio(root: Path, relative_path: str, *, seed: int, duplicate_from: Path | None = None) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if duplicate_from is not None:
        path.write_bytes(duplicate_from.read_bytes())
        return
    if path.suffix.casefold() not in {".wav", ".wave", ".flac", ".ogg", ".aiff", ".aif"}:
        path.write_bytes(b"unsupported fixture format")
        return
    rng = np.random.default_rng(seed)
    samples = (0.08 * rng.standard_normal(8000)).astype(np.float32)
    sf.write(path, samples, 16000, subtype="PCM_16")


def _write_metadata(root: Path, rows: list[dict[str, str]], *, corrupt_index: int | None = None) -> Path:
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for field in ("permission_evidence_ref", "label_source", "label_provenance"):
            if not row[field]:
                continue
            evidence_file = root / row[field]
            evidence_file.parent.mkdir(parents=True, exist_ok=True)
            evidence_file.write_text("Temporary unit-test fixture only; not project evidence.\n", encoding="utf-8")
    for index, row in enumerate(rows):
        relative_path = row["file_path"]
        if index == corrupt_index:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a valid WAV")
        else:
            _write_audio(root, relative_path, seed=index + 10)
    metadata_path = root / "labeled_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def _balanced_rows() -> list[dict[str, str]]:
    return [_metadata_row(index, label) for index, label in enumerate((0, 0, 0, 1, 1, 1))]


def test_assigns_source_and_speaker_disjoint_train_dev_eval_splits(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path, _balanced_rows())

    records, report = prepare_training_splits(metadata_path, seed=17)

    assert report.ready
    assert report.output_records == 6
    assert {row["split"] for row in records} == {"train", "dev", "eval"}
    assert all(report.class_counts[split][name] > 0 for split in ("train", "dev", "eval") for name in ("bona_fide", "spoof"))
    assert not validate_split_leakage(records)
    assert all(row["audio_sha256"] for row in records)

    output_path = tmp_path / "generated" / "split_manifest.csv"
    write_split_manifest(records, output_path)
    for split in ("train", "dev"):
        training_records = read_manifest_records(output_path, split)
        assert training_records
        assert {record.split for record in training_records} == {split}


def test_same_source_and_speaker_are_kept_in_one_group(tmp_path: Path) -> None:
    rows = [_metadata_row(index, index % 2, source_id="source_shared", speaker_id="speaker_shared") for index in range(6)]
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert any(finding.code == "insufficient_independent_groups" for finding in report.findings)
    assert len({row["split"] for row in records}) == 0


def test_audit_only_reports_classes_without_creating_partitions(tmp_path: Path) -> None:
    rows = [_metadata_row(index, label, source_id="podcast_source", speaker_id="speaker_shared") for index, label in enumerate((0, 0, 0))]
    metadata_path = _write_metadata(tmp_path, rows)

    report = audit_labeled_metadata(metadata_path)

    assert not report.ready
    assert report.input_class_counts == {"bona_fide": 3, "spoof": 0}
    assert report.class_counts["train"]["total"] == 0
    assert any(finding.code == "missing_class" for finding in report.findings)
    assert not list(tmp_path.glob("*split*.csv"))


def test_reports_duplicate_audio_content(tmp_path: Path) -> None:
    rows = _balanced_rows()
    metadata_path = _write_metadata(tmp_path, rows)
    duplicate_path = tmp_path / rows[0]["file_path"]
    _write_audio(tmp_path, rows[1]["file_path"], seed=99, duplicate_from=duplicate_path)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert report.duplicate_audio_files == 1
    assert any(finding.code == "duplicate_audio_content" for finding in report.findings)
    assert records == []


def test_reports_corrupt_and_missing_audio_files(tmp_path: Path) -> None:
    rows = _balanced_rows()
    metadata_path = _write_metadata(tmp_path, rows, corrupt_index=0)
    (tmp_path / rows[1]["file_path"]).unlink()

    records, report = prepare_training_splits(metadata_path)

    assert report.unreadable_audio_files == 1
    assert report.missing_audio_files == 1
    assert any(finding.code == "corrupt_audio" for finding in report.findings)
    assert any(finding.code == "missing_audio" for finding in report.findings)
    assert not report.ready
    assert records == []


def test_missing_class_is_reported_without_inventing_labels(tmp_path: Path) -> None:
    rows = [_metadata_row(index, 0) for index in range(4)]
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert any(finding.code == "missing_class" for finding in report.findings)
    assert records == []


def test_invalid_label_is_reported_without_guessing_from_metadata(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[0]["label"] = "synthetic"
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert any(finding.code == "invalid_label" for finding in report.findings)
    assert records == []


def test_spoof_label_requires_generation_and_label_provenance(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[3]["generation_method"] = "not_applicable"
    rows[3]["label_provenance"] = ""
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    codes = {finding.code for finding in report.findings}
    assert "missing_generation_evidence" in codes
    assert "missing_metadata" in codes
    assert not report.ready
    assert records == []


def test_ungranted_permission_blocks_training_readiness(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[0]["permission_status"] = "pending"
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert any(finding.code == "permission_not_granted" for finding in report.findings)
    assert records == []


def test_unsupported_audio_format_blocks_training_readiness(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[0]["file_path"] = "audio/clip_0.mp3"
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert any(finding.code == "unsupported_audio_format" for finding in report.findings)
    assert records == []


def test_unknown_language_and_class_imbalance_are_reported_separately(tmp_path: Path) -> None:
    rows = [_metadata_row(index, label) for index, label in enumerate((0, 0, 0, 0, 0, 0, 1, 1, 1))]
    rows[0]["language"] = "unknown"
    metadata_path = _write_metadata(tmp_path, rows)

    _records, report = prepare_training_splits(metadata_path, imbalance_warning_ratio=1.5)

    assert any(finding.code == "unknown_language" for finding in report.findings)
    assert any(finding.code == "class_imbalance" for finding in report.findings)
    assert report.imbalance_ratio == pytest.approx(2.0)


def test_direct_leakage_check_finds_source_and_speaker_overlap() -> None:
    rows = [
        {"source_id": "same_source", "speaker_id": "speaker_a", "split": "train"},
        {"source_id": "other_source", "speaker_id": "speaker_a", "split": "eval"},
        {"source_id": "same_source", "speaker_id": "speaker_b", "split": "dev"},
    ]

    findings = validate_split_leakage(rows)

    assert sum(item.code == "source_leakage" for item in findings) == 1
    assert sum(item.code == "speaker_leakage" for item in findings) == 1


def test_split_manifest_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_split_manifest([], output)

    assert output.read_text(encoding="utf-8") == "keep"


def test_preassigned_splits_are_rejected_instead_of_silently_repartitioned(tmp_path: Path) -> None:
    rows = _balanced_rows()
    rows[0]["split"] = "eval"
    metadata_path = _write_metadata(tmp_path, rows)

    records, report = prepare_training_splits(metadata_path)

    assert not report.ready
    assert any(finding.code == "preassigned_split" for finding in report.findings)
    assert records == []
