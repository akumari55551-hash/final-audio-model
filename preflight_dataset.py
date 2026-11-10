"""Read-only preflight checks for the local podcast media manifests."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

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
SUPPORTED_MEDIA_SUFFIXES = frozenset(
    {
        ".avi",
        ".flac",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".wav",
        ".webm",
    }
)
KNOWN_LANGUAGES = frozenset({"english", "hindi", "kannada", "tamil", "telugu"})
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Finding:
    """One actionable validation result."""

    severity: str
    code: str
    message: str
    source_id: str = ""
    path: str = ""


@dataclass(frozen=True)
class PreflightReport:
    """Summary of checks performed without changing files."""

    link_rows: int
    manifest_rows: int
    unique_source_ids: int
    granted_sources: int
    media_files_found: int
    unknown_language_sources: int
    findings: tuple[Finding, ...]

    @property
    def error_count(self) -> int:
        return sum(finding.severity == "ERROR" for finding in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(finding.severity == "WARNING" for finding in self.findings)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def format_text(self) -> str:
        """Return a human-readable report for stdout; no report file is written."""
        lines = [
            "DATASET PREFLIGHT (read-only)",
            "-" * 36,
            f"Podcast link records: {self.link_rows}",
            f"Processing manifest records: {self.manifest_rows}",
            f"Unique source IDs across files: {self.unique_source_ids}",
            f"Sources with granted permission in both files: {self.granted_sources}",
            f"Supported local media files found: {self.media_files_found}",
            f"Sources with unknown language labels: {self.unknown_language_sources}",
            f"Errors: {self.error_count}; warnings: {self.warning_count}",
        ]
        if self.findings:
            lines.extend(("", "Findings:"))
            for finding in self.findings:
                location = f" source_id={finding.source_id}" if finding.source_id else ""
                path = f" path={finding.path}" if finding.path else ""
                lines.append(
                    f"[{finding.severity}] {finding.code}:{location}{path} {finding.message}"
                )
        lines.extend(("", "Result: " + ("PASS" if self.ok else "FAIL")))
        return "\n".join(lines)


def _read_csv(
    path: Path,
    label: str,
    required_fields: tuple[str, ...],
    findings: list[Finding],
) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        findings.append(
            Finding("ERROR", "missing_csv", f"{label} CSV does not exist.", path=str(path))
        )
        return [], set()

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                findings.append(
                    Finding("ERROR", "duplicate_columns", f"{label} CSV has duplicate column names.", path=str(path))
                )
            missing_fields = sorted(set(required_fields) - set(headers))
            if missing_fields:
                findings.append(
                    Finding(
                        "ERROR",
                        "missing_columns",
                        f"{label} CSV is missing required columns: {', '.join(missing_fields)}.",
                        path=str(path),
                    )
                )
            rows = [
                {str(key): (value or "").strip() for key, value in row.items() if key is not None}
                for row in reader
            ]
            return rows, set(headers)
    except (OSError, UnicodeError, csv.Error) as error:
        findings.append(
            Finding("ERROR", "unreadable_csv", f"Could not read {label} CSV: {error}.", path=str(path))
        )
        return [], set()


def _validate_rows(
    rows: list[dict[str, str]],
    label: str,
    required_fields: tuple[str, ...],
    findings: list[Finding],
) -> tuple[dict[str, dict[str, str]], set[str], set[str]]:
    by_id: dict[str, dict[str, str]] = {}
    duplicate_ids: set[str] = set()
    invalid_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        source_id = row.get("source_id", "").strip()
        if not source_id or not SOURCE_ID_PATTERN.fullmatch(source_id):
            invalid_ids.add(source_id or f"row {row_number}")
            findings.append(
                Finding(
                    "ERROR",
                    "invalid_source_id",
                    f"{label} row {row_number} has a missing or invalid source ID.",
                    source_id=source_id,
                )
            )
        elif source_id in by_id:
            duplicate_ids.add(source_id)
            findings.append(
                Finding(
                    "ERROR",
                    "duplicate_source_id",
                    f"{label} contains source ID more than once (row {row_number}).",
                    source_id=source_id,
                )
            )
        else:
            by_id[source_id] = row

        for field in required_fields:
            value = row.get(field, "").strip()
            # audio_file is intentionally blank until authorized media is supplied.
            if not value and not (label == "Podcast links" and field == "audio_file"):
                if field == "language":
                    continue
                findings.append(
                    Finding(
                        "ERROR",
                        "missing_required_value",
                        f"{label} row {row_number} has an empty required field: {field}.",
                        source_id=source_id,
                    )
                )

        permission = row.get("permission_status", "").strip().casefold()
        if permission != "granted":
            findings.append(
                Finding(
                    "ERROR",
                    "invalid_permission",
                    f"{label} permission_status must be 'granted'; found {permission or '<blank>'!r}.",
                    source_id=source_id,
                )
            )

    return by_id, duplicate_ids, invalid_ids


def _resolve_local_path(raw_path: str, project_root: Path) -> Path | None:
    """Resolve a manifest path while preventing access outside the project root."""
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        return None
    return resolved


def _is_remote_url(value: str) -> bool:
    return urlsplit(value).scheme.casefold() in {"http", "https"}


def run_preflight(
    project_root: Path,
    links_path: Path | None = None,
    manifest_path: Path | None = None,
) -> PreflightReport:
    """Validate schemas, IDs, permissions, and local inputs without writing files."""
    project_root = project_root.resolve()
    dataset_dir = project_root / "PodcastDataset"
    links_path = links_path or dataset_dir / "podcast_links.csv"
    manifest_path = manifest_path or dataset_dir / "processing_manifest.csv"
    findings: list[Finding] = []

    link_rows, _ = _read_csv(links_path, "Podcast links", LINK_FIELDS, findings)
    manifest_rows, _ = _read_csv(manifest_path, "Processing manifest", MANIFEST_FIELDS, findings)
    links, duplicate_link_ids, invalid_link_ids = _validate_rows(
        link_rows, "Podcast links", LINK_FIELDS, findings
    )
    manifest, duplicate_manifest_ids, invalid_manifest_ids = _validate_rows(
        manifest_rows, "Processing manifest", MANIFEST_FIELDS, findings
    )

    link_ids = set(links)
    manifest_ids = set(manifest)
    for source_id in sorted(link_ids - manifest_ids):
        findings.append(
            Finding("ERROR", "missing_manifest_source", "Source is absent from processing manifest.", source_id)
        )
    for source_id in sorted(manifest_ids - link_ids):
        findings.append(
            Finding("ERROR", "missing_link_source", "Source is absent from podcast links.", source_id)
        )

    common_ids = link_ids & manifest_ids
    authorized_ids: set[str] = set()
    unknown_languages: dict[str, set[str]] = defaultdict(set)
    for label, rows in (("Podcast links", link_rows), ("Processing manifest", manifest_rows)):
        for row_number, row in enumerate(rows, start=2):
            language = row.get("language", "").strip()
            if language.casefold() not in KNOWN_LANGUAGES:
                source_id = row.get("source_id", "").strip() or f"{label} row {row_number}"
                unknown_languages[source_id].add(language or "<blank>")

    for source_id in common_ids:
        link_row = links[source_id]
        manifest_row = manifest[source_id]
        link_permission = link_row.get("permission_status", "").casefold()
        manifest_permission = manifest_row.get("permission_status", "").casefold()
        if link_permission != manifest_permission:
            findings.append(
                Finding(
                    "ERROR",
                    "permission_mismatch",
                    "Permission status differs between the two CSV files.",
                    source_id,
                )
            )
        if (
            link_permission == "granted"
            and manifest_permission == "granted"
            and source_id not in duplicate_link_ids
            and source_id not in duplicate_manifest_ids
            and source_id not in invalid_link_ids
            and source_id not in invalid_manifest_ids
        ):
            authorized_ids.add(source_id)

    for source_id, labels in sorted(unknown_languages.items()):
        findings.append(
            Finding(
                "WARNING",
                "unknown_language",
                f"Unknown or missing language label(s): {', '.join(sorted(labels))}.",
                source_id,
            )
        )

    media_by_source: dict[str, set[Path]] = defaultdict(set)
    unsupported_by_source: dict[str, set[Path]] = defaultdict(set)
    candidate_sources_by_path: dict[Path, set[str]] = defaultdict(set)
    media_sources_by_path: dict[Path, set[str]] = defaultdict(set)

    for source_id in sorted(authorized_ids):
        candidates = (
            ("audio_file", links[source_id].get("audio_file", "")),
            ("original_video", manifest[source_id].get("original_video", "")),
        )
        for field, raw_path in candidates:
            if not raw_path:
                continue
            if _is_remote_url(raw_path):
                findings.append(
                    Finding(
                        "WARNING",
                        "remote_url_not_local_media",
                        f"{field} contains a URL; URLs are not treated as local media.",
                        source_id,
                        raw_path,
                    )
                )
                continue

            resolved = _resolve_local_path(raw_path, project_root)
            if resolved is None:
                findings.append(
                    Finding(
                        "ERROR",
                        "media_path_outside_project",
                        f"{field} must resolve inside the project directory.",
                        source_id,
                        raw_path,
                    )
                )
                continue
            candidate_sources_by_path[resolved].add(source_id)
            if not resolved.exists():
                if resolved.suffix and resolved.suffix.casefold() not in SUPPORTED_MEDIA_SUFFIXES:
                    unsupported_by_source[source_id].add(resolved)
                continue

            if resolved.is_file():
                files = [resolved]
            elif resolved.is_dir():
                files = [path.resolve() for path in resolved.rglob("*") if path.is_file()]
            else:
                files = []

            for media_file in files:
                try:
                    media_file.relative_to(project_root)
                except ValueError:
                    findings.append(
                        Finding(
                            "ERROR",
                            "media_path_outside_project",
                            "Media file resolves outside the project directory.",
                            source_id,
                            str(media_file),
                        )
                    )
                    continue
                if media_file.suffix.casefold() not in SUPPORTED_MEDIA_SUFFIXES:
                    unsupported_by_source[source_id].add(media_file)
                    continue
                media_by_source[source_id].add(media_file)
                media_sources_by_path[media_file].add(source_id)

    for source_id in sorted(authorized_ids):
        if not media_by_source[source_id]:
            findings.append(
                Finding(
                    "ERROR",
                    "missing_local_media",
                    "No supported local media file was found for this authorized source.",
                    source_id,
                )
            )
        for unsupported in sorted(unsupported_by_source[source_id]):
            findings.append(
                Finding(
                    "ERROR",
                    "unsupported_media_format",
                    f"Unsupported local media extension: {unsupported.suffix or '<none>'}.",
                    source_id,
                    str(unsupported),
                )
            )

    all_path_sources: dict[Path, set[str]] = defaultdict(set)
    for path, source_ids in candidate_sources_by_path.items():
        all_path_sources[path].update(source_ids)
    for path, source_ids in media_sources_by_path.items():
        all_path_sources[path].update(source_ids)
    for media_file, source_ids in sorted(all_path_sources.items(), key=lambda item: str(item[0])):
        if len(source_ids) > 1:
            findings.append(
                Finding(
                    "ERROR",
                    "duplicate_media_path",
                    "The same local media file is assigned to multiple source IDs.",
                    source_id=", ".join(sorted(source_ids)),
                    path=str(media_file),
                )
            )

    unique_source_ids = link_ids | manifest_ids
    granted_sources = len(authorized_ids)
    media_files_found = len(media_sources_by_path)
    return PreflightReport(
        link_rows=len(link_rows),
        manifest_rows=len(manifest_rows),
        unique_source_ids=len(unique_source_ids),
        granted_sources=granted_sources,
        media_files_found=media_files_found,
        unknown_language_sources=len(unknown_languages),
        findings=tuple(findings),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only preflight for PodcastDataset CSVs and local media.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing PodcastDataset (default: this script's directory).",
    )
    args = parser.parse_args(argv)
    report = run_preflight(args.root)
    print(report.format_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())