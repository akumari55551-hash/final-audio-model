from __future__ import annotations

import csv
from pathlib import Path

import pytest

from preflight_dataset import run_preflight

LINK_FIELDS = ("source_id", "youtube_url", "language", "permission_status", "audio_file")
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


def _write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset(
    root: Path,
    *,
    source_ids: tuple[str, ...] = ("P001",),
    permission: str = "granted",
    language: str = "english",
    media_paths: tuple[str, ...] | None = None,
) -> None:
    dataset = root / "PodcastDataset"
    links: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    for index, source_id in enumerate(source_ids):
        media_path = (
            media_paths[index]
            if media_paths is not None
            else f"PodcastDataset/originals/{source_id}"
        )
        links.append(
            {
                "source_id": source_id,
                "youtube_url": f"https://www.youtube.com/watch?v=fixture-{index}",
                "language": language,
                "permission_status": permission,
                "audio_file": "",
            }
        )
        manifest.append(
            {
                "source_id": source_id,
                "youtube_url": links[-1]["youtube_url"],
                "language": language,
                "permission_status": permission,
                "original_video": media_path,
                "mp3_file": f"PodcastDataset/mp3_audio/{source_id}.mp3",
                "wav_folder": f"PodcastDataset/clips/{source_id}",
                "processing_status": "waiting_for_audio",
            }
        )
    _write_rows(dataset / "podcast_links.csv", LINK_FIELDS, links)
    _write_rows(dataset / "processing_manifest.csv", MANIFEST_FIELDS, manifest)


def test_valid_authorized_local_media_passes_without_writing_report(tmp_path: Path) -> None:
    media = tmp_path / "PodcastDataset" / "originals" / "P001" / "recording.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fixture media placeholder")
    _write_dataset(tmp_path)

    report = run_preflight(tmp_path)

    assert report.ok
    assert report.link_rows == 1
    assert report.manifest_rows == 1
    assert report.unique_source_ids == 1
    assert report.granted_sources == 1
    assert report.media_files_found == 1
    assert not list(tmp_path.rglob("*.json"))


def test_missing_local_media_is_reported_and_youtube_url_is_not_used(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    report = run_preflight(tmp_path)

    assert not report.ok
    assert any(finding.code == "missing_local_media" for finding in report.findings)
    assert not any(finding.code == "remote_url_not_local_media" for finding in report.findings)
    assert report.media_files_found == 0


def test_youtube_url_in_local_media_field_is_never_opened_as_a_file(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    links_path = tmp_path / "PodcastDataset" / "podcast_links.csv"
    with links_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    rows[0]["audio_file"] = rows[0]["youtube_url"]
    _write_rows(links_path, LINK_FIELDS, rows)

    report = run_preflight(tmp_path)

    assert any(finding.code == "remote_url_not_local_media" for finding in report.findings)
    assert any(finding.code == "missing_local_media" for finding in report.findings)
    assert report.media_files_found == 0


def test_duplicate_source_ids_are_reported(tmp_path: Path) -> None:
    _write_dataset(tmp_path, source_ids=("P001", "P001"))

    report = run_preflight(tmp_path)

    assert sum(finding.code == "duplicate_source_id" for finding in report.findings) == 2


def test_invalid_permission_does_not_authorize_media_inspection(tmp_path: Path) -> None:
    media = tmp_path / "PodcastDataset" / "originals" / "P001" / "recording.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fixture media placeholder")
    _write_dataset(tmp_path, permission="approved")

    report = run_preflight(tmp_path)

    assert any(finding.code == "invalid_permission" for finding in report.findings)
    assert report.granted_sources == 0
    assert report.media_files_found == 0


def test_unsupported_local_format_is_reported(tmp_path: Path) -> None:
    media = tmp_path / "PodcastDataset" / "originals" / "P001" / "recording.xyz"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"not an audio format")
    _write_dataset(tmp_path)

    report = run_preflight(tmp_path)

    assert any(finding.code == "unsupported_media_format" for finding in report.findings)
    assert any(finding.code == "missing_local_media" for finding in report.findings)


def test_unsupported_extension_is_reported_even_when_file_is_missing(tmp_path: Path) -> None:
    missing_media = "PodcastDataset/originals/P001/recording.xyz"
    _write_dataset(tmp_path, media_paths=(missing_media,))

    report = run_preflight(tmp_path)

    assert any(finding.code == "unsupported_media_format" for finding in report.findings)
    assert any(finding.code == "missing_local_media" for finding in report.findings)


def test_duplicate_media_path_and_invalid_source_id_are_reported(tmp_path: Path) -> None:
    shared = "PodcastDataset/originals/shared.wav"
    media = tmp_path / shared
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fixture media placeholder")
    _write_dataset(
        tmp_path,
        source_ids=("P001", "P002", "../P003"),
        media_paths=(shared, shared, "PodcastDataset/originals/P003"),
    )

    report = run_preflight(tmp_path)

    assert any(finding.code == "invalid_source_id" for finding in report.findings)
    assert any(finding.code == "duplicate_media_path" for finding in report.findings)


def test_unknown_language_is_a_separate_warning(tmp_path: Path) -> None:
    media = tmp_path / "PodcastDataset" / "originals" / "P001" / "recording.wav"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fixture media placeholder")
    _write_dataset(tmp_path, language="unknown")

    report = run_preflight(tmp_path)

    assert report.ok
    assert report.unknown_language_sources == 1
    assert report.warning_count == 1
    assert any(finding.code == "unknown_language" for finding in report.findings)


@pytest.mark.parametrize("missing_field", ["permission_status", "source_id"])
def test_missing_required_column_is_reported(tmp_path: Path, missing_field: str) -> None:
    _write_dataset(tmp_path)
    links_path = tmp_path / "PodcastDataset" / "podcast_links.csv"
    with links_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    fields = tuple(field for field in LINK_FIELDS if field != missing_field)
    rows_without_field = [{key: value for key, value in row.items() if key in fields} for row in rows]
    _write_rows(links_path, fields, rows_without_field)

    report = run_preflight(tmp_path)

    assert any(finding.code == "missing_columns" for finding in report.findings)