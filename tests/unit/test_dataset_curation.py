"""Temporary fixtures test curation mechanics, never project dataset facts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.dataset_curation import CURATION_FIELDS, curate_dataset, write_report

LINK_FIELDS = ("source_id", "youtube_url", "language", "permission_status", "audio_file")
PROCESSING_FIELDS = (
    "source_id",
    "youtube_url",
    "language",
    "permission_status",
    "original_video",
    "mp3_file",
    "wav_folder",
    "processing_status",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_registry(root: Path, count: int = 16) -> tuple[Path, Path, bytes, bytes]:
    dataset = root / "PodcastDataset"
    links = [
        {
            "source_id": f"P{index:03d}",
            "youtube_url": f"https://www.youtube.com/watch?v=test-record-{index}",
            "language": "unknown" if index == 1 else "english",
            "permission_status": "granted",
            "audio_file": "",
        }
        for index in range(1, count + 1)
    ]
    processing = [
        {
            "source_id": row["source_id"],
            "youtube_url": row["youtube_url"],
            "language": row["language"],
            "permission_status": row["permission_status"],
            "original_video": f"PodcastDataset/originals/{row['source_id']}",
            "mp3_file": f"PodcastDataset/mp3_audio/{row['source_id']}.mp3",
            "wav_folder": f"PodcastDataset/clips/{row['source_id']}",
            "processing_status": "waiting_for_audio",
        }
        for row in links
    ]
    links_path, processing_path = dataset / "podcast_links.csv", dataset / "processing_manifest.csv"
    _write_csv(links_path, LINK_FIELDS, links)
    _write_csv(processing_path, PROCESSING_FIELDS, processing)
    (dataset / "clip_metadata.csv").write_text(
        "clip_id,source_id,file_path,duration,sample_rate,language,speaker_id,quality_status\n", encoding="utf-8"
    )
    return links_path, processing_path, links_path.read_bytes(), processing_path.read_bytes()


def _write_audio(root: Path, relative: str, seed: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = (0.08 * rng.standard_normal(8000)).astype(np.float32)
    sf.write(path, samples, 16000, subtype="PCM_16")
    return path


def _curation_row(root: Path, index: int, label: int) -> dict[str, str]:
    path = _write_audio(root, f"curated/clip_{index}.wav", index + 10)
    evidence_dir = root / "curation" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    permission_ref = f"evidence/permission-{index}.md"
    label_source_ref = f"evidence/label-source-{index}.md"
    label_provenance_ref = f"evidence/label-review-{index}.md"
    for evidence_ref in (permission_ref, label_source_ref, label_provenance_ref):
        (evidence_dir / Path(evidence_ref).name).write_text(
            "Temporary test fixture only; not project dataset evidence.\n", encoding="utf-8"
        )
    return {
        "audio_id": f"audio-{index}",
        "source_id": f"source-{index}",
        "speaker_id": f"speaker-{index}",
        "audio_path": (Path("..") / path.relative_to(root)).as_posix(),
        "binary_label": str(label),
        "language": "english",
        "permission_status": "granted",
        "permission_evidence_ref": permission_ref,
        "label_source": label_source_ref,
        "label_provenance": label_provenance_ref,
        "synthetic_generation_method": "fixture_tts" if label == 1 else "not_applicable",
        "split_assignment": "unassigned",
    }


def _write_curation_csv(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "curation" / "curation_metadata.csv"
    _write_csv(path, CURATION_FIELDS, rows)
    return path


def test_integrates_16_podcast_records_as_pending_without_modifying_them(tmp_path: Path) -> None:
    _links, _manifest, links_before, manifest_before = _write_registry(tmp_path)

    report, generated = curate_dataset(tmp_path)

    assert report.podcast_source_count == 16
    assert report.unique_podcast_source_count == 16
    assert report.pending_count == 16
    assert report.accepted_count == 0
    assert report.rejected_count == 0
    assert not report.training_ready
    assert generated == []
    assert all(not item.details.get("youtube_url_used_as_media", True) for item in report.pending)
    assert _links.read_bytes() == links_before
    assert _manifest.read_bytes() == manifest_before


def test_reports_local_wav_technical_metadata_without_assigning_labels_or_identity(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    first = _write_audio(tmp_path, "data/raw/genuine/spk001/spk001_1.wav", 1)
    _write_audio(tmp_path, "data/raw/genuine/spk001/spk001_2.wav", 2)

    report, generated = curate_dataset(tmp_path)

    raw_items = [item for item in report.pending if item.audio_path.startswith("data/raw/")]
    assert len(raw_items) == 2
    assert all(item.source_id == "" for item in raw_items)
    assert all(item.details["technical_audio"]["sample_rate_hz"] == 16000 for item in raw_items)
    assert all(item.details["technical_audio"]["channels"] == 1 for item in raw_items)
    assert all(item.details["technical_audio"]["duration_seconds"] == 0.5 for item in raw_items)
    assert all("label_not_verified" in item.reason_codes for item in raw_items)
    assert generated == []
    second = tmp_path / "data/raw/genuine/spk001/spk001_2.wav"
    assert first.is_file() and second.is_file()


def test_legacy_clip_metadata_audio_is_pending_and_keeps_source_association(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=1)
    clip = _write_audio(tmp_path, "PodcastDataset/clips/P001/P001_0000.wav", 6)
    clip_metadata = tmp_path / "PodcastDataset" / "clip_metadata.csv"
    _write_csv(
        clip_metadata,
        ("clip_id", "source_id", "file_path", "duration", "sample_rate", "language", "speaker_id", "quality_status"),
        [
            {
                "clip_id": "P001_0000",
                "source_id": "P001",
                "file_path": clip.relative_to(tmp_path).as_posix(),
                "duration": "0.5",
                "sample_rate": "16000",
                "language": "english",
                "speaker_id": "",
                "quality_status": "PASS",
            }
        ],
    )

    report, generated = curate_dataset(tmp_path)

    assert report.clip_metadata_row_count == 1
    assert any(item.source_id == "P001" and item.audio_path.endswith("P001_0000.wav") for item in report.pending)
    assert report.accepted_count == 0
    assert generated == []


def test_curation_requires_label_and_permission_evidence_and_never_guesses(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    row["permission_evidence_ref"] = ""
    row["label_provenance"] = "unverified"
    path = _write_curation_csv(tmp_path, [row])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert {"permission_evidence_missing", "label_provenance_missing"}.issubset(
        set(report.rejected[0].reason_codes)
    )
    assert generated == []


def test_non_granted_permission_is_rejected(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    row["permission_status"] = "pending"
    path = _write_curation_csv(tmp_path, [row])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.rejected_count == 1
    assert "permission_not_granted" in report.rejected[0].reason_codes
    assert generated == []


def test_external_evidence_is_pending_and_never_fetched(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    row["label_provenance"] = "https://example.invalid/label-evidence"
    path = _write_curation_csv(tmp_path, [row])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.pending_count == 1
    assert report.pending[0].reason_codes == ("external_evidence_not_locally_verified",)
    assert report.accepted_count == 0
    assert generated == []


def test_empty_local_evidence_document_is_rejected(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    evidence_path = tmp_path / "curation" / row["permission_evidence_ref"]
    evidence_path.write_text("", encoding="utf-8")
    path = _write_curation_csv(tmp_path, [row])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.rejected_count == 1
    assert "permission_evidence_ref_document_empty" in report.rejected[0].reason_codes
    assert generated == []


def test_unknown_label_is_pending_and_not_admitted(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    row["binary_label"] = ""
    path = _write_curation_csv(tmp_path, [row])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.pending_count == 1
    assert report.pending[0].reason_codes == ("label_unassigned",)
    assert report.accepted_count == 0
    assert generated == []


def test_unverified_label_value_is_rejected(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    row["binary_label"] = "genuine"
    path = _write_curation_csv(tmp_path, [row])

    report, _ = curate_dataset(tmp_path, curation_csv=path)

    assert report.rejected_count == 1
    assert "invalid_binary_label" in report.rejected[0].reason_codes


def test_spoof_requires_documented_generation_method(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 1)
    row["synthetic_generation_method"] = "unknown"
    path = _write_curation_csv(tmp_path, [row])

    report, _ = curate_dataset(tmp_path, curation_csv=path)

    assert report.rejected_count == 1
    assert "synthetic_generation_method_missing" in report.rejected[0].reason_codes


def test_remote_url_is_never_used_as_local_audio(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    row = _curation_row(tmp_path, 1, 0)
    row["audio_path"] = "https://example.invalid/audio.wav"
    path = _write_curation_csv(tmp_path, [row])

    report, _ = curate_dataset(tmp_path, curation_csv=path)

    assert report.rejected_count == 1
    assert "remote_url_not_local_media" in report.rejected[0].reason_codes


def test_duplicate_audio_content_rejects_all_duplicate_curations(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    first = _curation_row(tmp_path, 1, 0)
    duplicate_path = _write_audio(tmp_path, "curated/duplicate.wav", 11)
    second = _curation_row(tmp_path, 2, 1)
    duplicate_path.write_bytes((tmp_path / "curated/clip_1.wav").read_bytes())
    second["audio_path"] = (Path("..") / duplicate_path.relative_to(tmp_path)).as_posix()
    path = _write_curation_csv(tmp_path, [first, second])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.accepted_count == 0
    assert report.rejected_count == 2
    assert all("duplicate_audio_content" in item.reason_codes for item in report.rejected)
    assert generated == []


def test_duplicate_audio_ids_reject_all_conflicting_rows(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    first = _curation_row(tmp_path, 1, 0)
    second = _curation_row(tmp_path, 2, 1)
    second["audio_id"] = first["audio_id"]
    path = _write_curation_csv(tmp_path, [first, second])

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.accepted_count == 0
    assert report.rejected_count == 2
    assert all("duplicate_audio_id" in item.reason_codes for item in report.rejected)
    assert generated == []


def test_curation_report_refuses_to_overwrite(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    report, _ = curate_dataset(tmp_path)
    output = tmp_path / "curation_report.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report(report, output)
    assert output.read_text(encoding="utf-8") == "keep"


def test_sufficient_evidenced_groups_generate_disjoint_trainer_manifest(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    rows = [_curation_row(tmp_path, index, label) for index, label in enumerate((0, 0, 0, 1, 1, 1))]
    path = _write_curation_csv(tmp_path, rows)

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.training_ready
    assert report.accepted_count == 6
    assert report.class_counts == {"bona_fide": 3, "spoof": 3}
    assert {row["split"] for row in generated} == {"train", "dev", "eval"}
    for label in (0, 1):
        for split in ("train", "dev", "eval"):
            assert any(int(row["label"]) == label and row["split"] == split for row in generated)


def test_not_enough_independent_groups_blocks_manifest_generation(tmp_path: Path) -> None:
    _write_registry(tmp_path, count=0)
    rows = [_curation_row(tmp_path, index, label) for index, label in enumerate((0, 0, 0, 1, 1, 1))]
    for row in rows:
        row["source_id"] = "same-source"
        row["speaker_id"] = "same-speaker"
    path = _write_curation_csv(tmp_path, rows)

    report, generated = curate_dataset(tmp_path, curation_csv=path)

    assert report.accepted_count == 6
    assert not report.training_ready
    assert generated == []
    assert any(item.code == "insufficient_independent_groups" for item in report.findings)
