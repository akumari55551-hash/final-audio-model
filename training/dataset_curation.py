"""Evidence-backed curation for local speech audio.

This module integrates the existing podcast registry without editing it. URLs
are metadata only and are never opened. Local files are inspected read-only;
labels are accepted only from a separately curated CSV with explicit evidence.
The CNN is not trained here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import soundfile as sf

from training.dataset_readiness import (
    CLASS_NAMES,
    DatasetReadinessError,
    ReadinessFinding,
    _source_speaker_components,
    prepare_training_splits,
    write_split_manifest,
)

CURATION_FIELDS = (
    "audio_id",
    "source_id",
    "speaker_id",
    "audio_path",
    "binary_label",
    "language",
    "permission_status",
    "permission_evidence_ref",
    "label_source",
    "label_provenance",
    "synthetic_generation_method",
    "split_assignment",
)
AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".aiff", ".aif"}
PLACEHOLDER_EVIDENCE = {"", "unknown", "unverified", "none", "n/a", "na", "pending"}
VALID_SPLIT_PLACEHOLDERS = {"", "unassigned", "pending"}


@dataclass(frozen=True)
class CurationItem:
    status: str
    audio_id: str
    source_id: str
    audio_path: str
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class DatasetCurationReport:
    podcast_source_count: int
    unique_podcast_source_count: int
    clip_metadata_row_count: int
    local_audio_count: int
    curation_input_count: int
    accepted_count: int
    rejected_count: int
    pending_count: int
    training_ready: bool
    class_counts: Mapping[str, int]
    independent_source_speaker_groups_by_class: Mapping[str, int]
    accepted: tuple[CurationItem, ...]
    rejected: tuple[CurationItem, ...]
    pending: tuple[CurationItem, ...]
    findings: tuple[ReadinessFinding, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_version": 1,
            "status": "ready_for_split_manifest" if self.training_ready else "not_ready",
            "training_ready": self.training_ready,
            "training_performed": False,
            "podcast_source_count": self.podcast_source_count,
            "unique_podcast_source_count": self.unique_podcast_source_count,
            "clip_metadata_row_count": self.clip_metadata_row_count,
            "local_audio_count": self.local_audio_count,
            "curation_input_count": self.curation_input_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "pending_count": self.pending_count,
            "class_counts": dict(self.class_counts),
            "independent_source_speaker_groups_by_class": dict(self.independent_source_speaker_groups_by_class),
            "accepted": [asdict(item) for item in self.accepted],
            "rejected": [asdict(item) for item in self.rejected],
            "pending": [asdict(item) for item in self.pending],
            "findings": [asdict(item) for item in self.findings],
            "interpretation": (
                "Accepted means required metadata/evidence references were supplied and technical checks passed. "
                "References still require human/documentary verification. No labels are inferred and no model was trained."
            ),
        }

    def format_text(self) -> str:
        lines = [
            "EVIDENCE-BACKED DATASET CURATION",
            "-" * 36,
            f"Podcast sources: {self.podcast_source_count} ({self.unique_podcast_source_count} unique IDs)",
            f"Existing clip metadata rows: {self.clip_metadata_row_count}",
            f"Local audio inspected: {self.local_audio_count}",
            f"Curation rows: {self.curation_input_count}",
            f"Accepted: {self.accepted_count}; rejected: {self.rejected_count}; pending: {self.pending_count}",
            f"Accepted class counts: {dict(self.class_counts)}",
            f"Independent source-speaker groups/class: {dict(self.independent_source_speaker_groups_by_class)}",
            f"Training data ready: {self.training_ready}",
            "No training or label inference was performed.",
        ]
        for item in (*self.rejected, *self.pending):
            path = f" path={item.audio_path}" if item.audio_path else ""
            reasons = ",".join(item.reason_codes) if item.reason_codes else item.details.get("message", "")
            lines.append(f"[{item.status}] {item.audio_id or item.source_id}{path}: {reasons}")
        for finding in self.findings:
            location = f" row={finding.row_number}" if finding.row_number is not None else ""
            lines.append(f"[{finding.severity}] {finding.code}:{location} {finding.message}")
        return "\n".join(lines)


class CurationError(ValueError):
    """Raised when input CSV structure cannot be read safely."""


def _read_csv(path: Path, required: Sequence[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required source CSV not found: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            missing = sorted(set(required) - set(headers))
            if missing:
                raise CurationError(f"{path}: missing required columns: {', '.join(missing)}")
            rows = [
                {str(key): (value or "").strip() for key, value in row.items() if key is not None}
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise CurationError(f"Could not read {path}: {error}") from error
    return rows, headers


def _resolve_local_path(raw: str, root: Path, base: Path) -> Path | None:
    if not raw or urlsplit(raw).scheme.casefold() in {"http", "https"}:
        return None
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _technical_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with sf.SoundFile(path, mode="r") as audio:
        samplerate = int(audio.samplerate)
        channels = int(audio.channels)
        frames = int(audio.frames)
        format_name = str(audio.format)
        subtype = str(audio.subtype)
        if samplerate <= 0 or channels <= 0 or frames <= 0:
            raise ValueError("empty_audio_or_invalid_header")
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            digest.update(block.tobytes())
    return {
        "sample_rate_hz": samplerate,
        "channels": channels,
        "duration_seconds": round(frames / samplerate, 6),
        "format": format_name,
        "subtype": subtype,
        "audio_sha256": digest.hexdigest(),
    }


def _candidate_paths(
    root: Path,
    podcast_rows: Sequence[dict[str, str]],
    processing_rows: Sequence[dict[str, str]],
    clip_rows: Sequence[dict[str, str]],
) -> tuple[dict[Path, set[str]], list[ReadinessFinding]]:
    paths: dict[Path, set[str]] = defaultdict(set)
    findings: list[ReadinessFinding] = []
    for row in podcast_rows:
        source_id = row.get("source_id", "")
        raw_path = row.get("audio_file", "")
        path = _resolve_local_path(raw_path, root, root) if raw_path else None
        if path is not None:
            if path.is_file():
                if source_id:
                    paths[path].add(source_id)
                else:
                    paths[path]
            elif path.exists():
                findings.append(ReadinessFinding("WARNING", "audio_file_not_file", "audio_file is not a local file; URL values are never fetched.", file_path=raw_path))
    for row in processing_rows:
        source_id = row.get("source_id", "")
        for field_name in ("original_video", "mp3_file", "wav_folder"):
            raw_path = row.get(field_name, "")
            path = _resolve_local_path(raw_path, root, root) if raw_path else None
            if path is None or not path.exists():
                continue
            if path.is_file():
                if path.suffix.casefold() in AUDIO_SUFFIXES:
                    if source_id:
                        paths[path].add(source_id)
                    else:
                        paths[path]
            elif path.is_dir() and field_name in {"original_video", "wav_folder"}:
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.casefold() in AUDIO_SUFFIXES:
                        if source_id:
                            paths[child.resolve()].add(source_id)
                        else:
                            paths[child.resolve()]
    for row in clip_rows:
        raw_path = row.get("file_path", "")
        path = _resolve_local_path(raw_path, root, root) if raw_path else None
        if path is None or path.suffix.casefold() not in AUDIO_SUFFIXES:
            continue
        if path.is_file():
            source_id = row.get("source_id", "")
            if source_id:
                paths[path].add(source_id)
            else:
                paths[path]
    raw_dir = root / "data" / "raw"
    if raw_dir.is_dir():
        for path in raw_dir.rglob("*"):
            if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES:
                paths[path.resolve()]  # Source/speaker identity is deliberately not inferred from directory names.
    return paths, findings


def _pending_source_items(
    podcast_rows: Sequence[dict[str, str]],
    processing_by_id: Mapping[str, dict[str, str]],
    source_paths: Mapping[Path, set[str]],
    root: Path,
) -> list[CurationItem]:
    items: list[CurationItem] = []
    paths_by_source: dict[str, list[Path]] = defaultdict(list)
    for path, source_ids in source_paths.items():
        for source_id in source_ids:
            if source_id:
                paths_by_source[source_id].append(path)

    for row in podcast_rows:
        source_id = row.get("source_id", "")
        local_paths = sorted(paths_by_source.get(source_id, []), key=lambda path: str(path).casefold())
        processing = processing_by_id.get(source_id, {})
        if not local_paths:
            items.append(
                CurationItem(
                    "PENDING",
                    "",
                    source_id,
                    "",
                    ("local_media_missing", "label_not_verified", "permission_evidence_missing"),
                    {
                        "language": row.get("language", "unknown"),
                        "permission_status": row.get("permission_status", ""),
                        "permission_evidence_ref": "",
                        "youtube_url_used_as_media": False,
                        "processing_status": processing.get("processing_status", ""),
                    },
                )
            )
            continue
        for path in local_paths:
            try:
                technical = _technical_metadata(path)
            except (OSError, RuntimeError, sf.SoundFileRuntimeError, ValueError) as error:
                items.append(
                    CurationItem("REJECTED", path.stem, source_id, path.relative_to(root).as_posix(), ("corrupt_audio",), {"error": type(error).__name__})
                )
                continue
            items.append(
                CurationItem(
                    "PENDING",
                    f"audio_{technical['audio_sha256'][:12]}",
                    source_id,
                    path.relative_to(root).as_posix(),
                    ("label_not_verified", "permission_evidence_missing", "speaker_identity_not_curated"),
                    {
                        "language": row.get("language", "unknown"),
                        "permission_status": row.get("permission_status", ""),
                        "permission_evidence_ref": "",
                        "technical_audio": technical,
                    },
                )
            )
    return items


def _raw_audio_items(source_paths: Mapping[Path, set[str]], root: Path) -> tuple[list[CurationItem], list[ReadinessFinding]]:
    items: list[CurationItem] = []
    findings: list[ReadinessFinding] = []
    for path, source_ids in sorted(source_paths.items(), key=lambda item: str(item[0]).casefold()):
        if source_ids:
            continue
        try:
            technical = _technical_metadata(path)
        except (OSError, RuntimeError, sf.SoundFileRuntimeError, ValueError) as error:
            items.append(
                CurationItem("REJECTED", path.stem, "", path.relative_to(root).as_posix(), ("corrupt_audio",), {"error": type(error).__name__})
            )
            findings.append(ReadinessFinding("ERROR", "corrupt_local_audio", f"Could not decode local audio ({type(error).__name__}).", file_path=path.relative_to(root).as_posix()))
            continue
        items.append(
            CurationItem(
                "PENDING",
                f"audio_{technical['audio_sha256'][:12]}",
                "",
                path.relative_to(root).as_posix(),
                ("source_identity_unverified", "speaker_identity_unverified", "label_not_verified", "permission_evidence_missing"),
                {"technical_audio": technical, "directory_name_not_used_as_label_or_identity": True},
            )
        )
    return items, findings


def _normalize_curation_row(row: Mapping[str, str], row_number: int, curation_base: Path, root: Path) -> tuple[dict[str, str] | None, list[str], bool]:
    errors: list[str] = []
    label_text = (row.get("binary_label") or "").strip()
    if not label_text:
        return None, ["label_unassigned"], True
    if label_text not in {"0", "1"}:
        errors.append("invalid_binary_label")
        label = ""
    else:
        label = label_text

    required_values = ("audio_id", "source_id", "speaker_id", "audio_path", "language", "permission_status", "permission_evidence_ref", "label_source", "label_provenance")
    for field_name in required_values:
        if not (row.get(field_name) or "").strip():
            errors.append(f"missing_{field_name}")
    permission = (row.get("permission_status") or "").strip().casefold()
    permission_ref = (row.get("permission_evidence_ref") or "").strip()
    if permission != "granted":
        errors.append("permission_not_granted")
    if permission_ref.casefold() in PLACEHOLDER_EVIDENCE:
        errors.append("permission_evidence_missing")
    if (row.get("label_source") or "").strip().casefold() in PLACEHOLDER_EVIDENCE:
        errors.append("label_source_missing")
    if (row.get("label_provenance") or "").strip().casefold() in PLACEHOLDER_EVIDENCE:
        errors.append("label_provenance_missing")
    external_evidence_refs: list[str] = []
    normalized_evidence_refs: dict[str, str] = {}
    for field_name in ("permission_evidence_ref", "label_source", "label_provenance"):
        reference = (row.get(field_name) or "").strip()
        if not reference or reference.casefold() in PLACEHOLDER_EVIDENCE:
            continue
        if urlsplit(reference).scheme.casefold() in {"http", "https"}:
            external_evidence_refs.append(field_name)
            normalized_evidence_refs[field_name] = reference
            continue
        evidence_path = _resolve_local_path(reference, root, curation_base)
        if evidence_path is None:
            errors.append(f"{field_name}_outside_project")
        elif not evidence_path.is_file():
            errors.append(f"{field_name}_document_missing")
        elif evidence_path.stat().st_size == 0:
            errors.append(f"{field_name}_document_empty")
        else:
            normalized_evidence_refs[field_name] = str(evidence_path)

    split_assignment = (row.get("split_assignment") or "").strip().casefold()
    if split_assignment not in VALID_SPLIT_PLACEHOLDERS:
        errors.append("preassigned_split_not_allowed")

    generation_method = (row.get("synthetic_generation_method") or "").strip()
    if label == "1" and generation_method.casefold() in PLACEHOLDER_EVIDENCE:
        errors.append("synthetic_generation_method_missing")
    if label == "0" and generation_method.casefold() not in PLACEHOLDER_EVIDENCE | {
        "not_applicable",
        "not applicable",
    }:
        errors.append("generation_method_not_applicable_for_bona_fide")

    raw_path = (row.get("audio_path") or "").strip()
    if urlsplit(raw_path).scheme.casefold() in {"http", "https"}:
        errors.append("remote_url_not_local_media")
        path = None
    else:
        path = _resolve_local_path(raw_path, root, curation_base)
        if path is None:
            errors.append("audio_path_outside_project_or_missing")
        elif path.suffix.casefold() not in AUDIO_SUFFIXES:
            errors.append("unsupported_audio_format")
        elif not path.is_file():
            errors.append("audio_file_missing")

    if errors or path is None:
        return None, sorted(set(errors)), False
    if external_evidence_refs:
        return None, ["external_evidence_not_locally_verified"], True

    normalized = {
        "file_path": str(path),
        "label": label,
        "split": "",
        "source_id": (row.get("source_id") or "").strip(),
        "source": (row.get("source_id") or "").strip(),
        "speaker_id": (row.get("speaker_id") or "").strip(),
        "utterance_id": (row.get("audio_id") or "").strip(),
        "language": (row.get("language") or "").strip(),
        "permission_status": permission,
        "permission_evidence_ref": normalized_evidence_refs.get("permission_evidence_ref", permission_ref),
        "generation_method": generation_method if label == "1" else "not_applicable",
        "label_source": normalized_evidence_refs.get("label_source", (row.get("label_source") or "").strip()),
        "label_provenance": normalized_evidence_refs.get(
            "label_provenance", (row.get("label_provenance") or "").strip()
        ),
        "attack_type": generation_method if label == "1" else "",
        "_audio_id": (row.get("audio_id") or "").strip(),
        "_synthetic_generation_method": generation_method,
        "_external_evidence_refs": ";".join(external_evidence_refs),
    }
    return normalized, [], False


def _to_readiness_csv_row(row: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def curate_dataset(
    project_root: Path | str,
    *,
    curation_csv: Path | str | None = None,
) -> tuple[DatasetCurationReport, list[dict[str, str]]]:
    """Merge the existing podcast registry with local evidence-curated rows.

    Existing CSVs/audio are read-only. Podcast records become pending until a
    local file and documentary label/permission evidence are supplied.
    """
    root = Path(project_root).expanduser().resolve()
    dataset_dir = root / "PodcastDataset"
    podcast_rows, _ = _read_csv(dataset_dir / "podcast_links.csv", ("source_id", "youtube_url", "language", "permission_status", "audio_file"))
    processing_rows, _ = _read_csv(dataset_dir / "processing_manifest.csv", ("source_id", "youtube_url", "language", "permission_status", "original_video", "mp3_file", "wav_folder", "processing_status"))
    clip_rows, _ = _read_csv(
        dataset_dir / "clip_metadata.csv",
        ("clip_id", "source_id", "file_path", "duration", "sample_rate", "language", "speaker_id", "quality_status"),
    )
    processing_by_id: dict[str, dict[str, str]] = {}
    findings: list[ReadinessFinding] = []
    for row_number, row in enumerate(processing_rows, start=2):
        source_id = row.get("source_id", "")
        if not source_id or source_id in processing_by_id:
            findings.append(ReadinessFinding("ERROR", "duplicate_or_missing_processing_source_id", "Processing manifest source IDs must be unique and non-empty.", row_number))
        else:
            processing_by_id[source_id] = row
    source_ids = [row.get("source_id", "") for row in podcast_rows]
    source_counts = Counter(source_ids)
    duplicates = {source_id for source_id, count in source_counts.items() if not source_id or count > 1}
    for source_id in sorted(duplicates):
        findings.append(ReadinessFinding("ERROR", "duplicate_or_missing_podcast_source_id", "Podcast source ID is missing or duplicated; the original row is preserved and excluded from curation.", file_path=source_id))
    podcast_id_set = set(source_ids) - {""}
    processing_id_set = set(processing_by_id)
    for source_id in sorted(podcast_id_set - processing_id_set):
        findings.append(ReadinessFinding("ERROR", "missing_processing_source", "Podcast source is absent from processing manifest.", file_path=source_id))
    for source_id in sorted(processing_id_set - podcast_id_set):
        findings.append(ReadinessFinding("ERROR", "orphan_processing_source", "Processing source is absent from podcast links.", file_path=source_id))
    link_by_id = {row.get("source_id", ""): row for row in podcast_rows if row.get("source_id")}
    for source_id in sorted(podcast_id_set & processing_id_set):
        if link_by_id[source_id].get("permission_status", "").casefold() != processing_by_id[source_id].get(
            "permission_status", ""
        ).casefold():
            findings.append(ReadinessFinding("ERROR", "permission_status_mismatch", "Permission status differs between the existing podcast CSVs.", file_path=source_id))

    source_paths, discovery_findings = _candidate_paths(root, podcast_rows, processing_rows, clip_rows)
    findings.extend(discovery_findings)
    pending = _pending_source_items(podcast_rows, processing_by_id, source_paths, root)
    local_items, local_findings = _raw_audio_items(source_paths, root)
    findings.extend(local_findings)
    pending.extend(local_items)

    curation_rows: list[dict[str, str]] = []
    curation_base = root
    if curation_csv is not None:
        curation_path = Path(curation_csv).expanduser()
        curation_path = curation_path.resolve() if curation_path.is_absolute() else (root / curation_path).resolve()
        curation_rows, headers = _read_csv(curation_path, CURATION_FIELDS)
        curation_base = curation_path.parent
        if len(headers) != len(set(headers)):
            findings.append(ReadinessFinding("ERROR", "duplicate_curation_columns", "Curation CSV has duplicate column names."))

    normalized_rows: list[dict[str, str]] = []
    rejected: list[CurationItem] = []
    curation_pending: list[CurationItem] = []
    for row_number, row in enumerate(curation_rows, start=2):
        normalized, errors, is_pending = _normalize_curation_row(row, row_number, curation_base, root)
        audio_id = (row.get("audio_id") or "").strip()
        source_id = (row.get("source_id") or "").strip()
        audio_path = (row.get("audio_path") or "").strip()
        if is_pending:
            curation_pending.append(CurationItem("PENDING", audio_id, source_id, audio_path, tuple(errors)))
        elif errors or normalized is None:
            rejected.append(CurationItem("REJECTED", audio_id, source_id, audio_path, tuple(errors)))
        else:
            normalized_rows.append(normalized)

    # Hash/decode accepted candidates once; reject every member of any duplicate group.
    hash_groups: dict[str, list[int]] = defaultdict(list)
    path_groups: dict[Path, list[int]] = defaultdict(list)
    audio_id_groups: dict[str, list[int]] = defaultdict(list)
    technical_errors: dict[int, str] = {}
    for index, row in enumerate(normalized_rows):
        path = Path(row["file_path"])
        path_groups[path].append(index)
        audio_id_groups[row["_audio_id"]].append(index)
        try:
            technical = _technical_metadata(path)
            row["audio_sha256"] = technical["audio_sha256"]
            hash_groups[technical["audio_sha256"]].append(index)
        except (OSError, RuntimeError, sf.SoundFileRuntimeError, ValueError) as error:
            technical_errors[index] = type(error).__name__

    duplicate_indexes: dict[int, str] = {}
    for indexes in audio_id_groups.values():
        if len(indexes) > 1:
            for index in indexes:
                duplicate_indexes[index] = "duplicate_audio_id"
    for indexes in path_groups.values():
        if len(indexes) > 1:
            for index in indexes:
                duplicate_indexes[index] = "duplicate_audio_path"
    for indexes in hash_groups.values():
        if len(indexes) > 1:
            for index in indexes:
                duplicate_indexes[index] = "duplicate_audio_content"

    accepted_rows: list[dict[str, str]] = []
    accepted: list[CurationItem] = []
    for index, row in enumerate(normalized_rows):
        if index in technical_errors:
            rejected.append(CurationItem("REJECTED", row["_audio_id"], row["source_id"], Path(row["file_path"]).relative_to(root).as_posix(), ("corrupt_audio",), {"error": technical_errors[index]}))
            continue
        if index in duplicate_indexes:
            rejected.append(CurationItem("REJECTED", row["_audio_id"], row["source_id"], Path(row["file_path"]).relative_to(root).as_posix(), (duplicate_indexes[index],)))
            continue
        accepted_rows.append(row)
        accepted.append(
            CurationItem(
                "ACCEPTED",
                row["_audio_id"],
                row["source_id"],
                Path(row["file_path"]).relative_to(root).as_posix(),
                (),
                {
                    "binary_label": int(row["label"]),
                    "speaker_id": row["speaker_id"],
                    "language": row["language"],
                    "permission_evidence_ref": row["permission_evidence_ref"],
                    "label_source": row["label_source"],
                    "label_provenance": row["label_provenance"],
                    "synthetic_generation_method": row["_synthetic_generation_method"],
                    "external_evidence_refs_not_fetched": row["_external_evidence_refs"],
                },
            )
        )

    split_rows: list[dict[str, str]] = []
    class_counts = {CLASS_NAMES[label]: 0 for label in CLASS_NAMES}
    groups_by_class = {CLASS_NAMES[label]: 0 for label in CLASS_NAMES}
    training_ready = False
    if accepted_rows:
        for row in accepted_rows:
            class_counts[CLASS_NAMES[int(row["label"])]] += 1
        try:
            with tempfile.TemporaryDirectory(prefix="dataset-curation-") as temp_dir:
                temp_input = Path(temp_dir) / "curated.csv"
                readiness_fields = tuple(key for key in accepted_rows[0] if not key.startswith("_"))
                with temp_input.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=readiness_fields)
                    writer.writeheader()
                    writer.writerows(_to_readiness_csv_row(row) for row in accepted_rows)
                split_rows, readiness_report = prepare_training_splits(temp_input, seed=42)
                findings.extend(readiness_report.findings)
                training_ready = readiness_report.ready
        except (DatasetReadinessError, OSError, ValueError) as error:
            findings.append(ReadinessFinding("ERROR", "split_generation_blocked", str(error)))
        components = _source_speaker_components(accepted_rows)
        for label in CLASS_NAMES:
            groups_by_class[CLASS_NAMES[label]] = sum(
                any(int(accepted_rows[index]["label"]) == label for index in component)
                for component in components
            )

    podcast_pending = [item for item in pending if item.source_id in set(source_ids)]
    report = DatasetCurationReport(
        podcast_source_count=len(podcast_rows),
        unique_podcast_source_count=len(set(source_ids) - duplicates),
        clip_metadata_row_count=len(clip_rows),
        local_audio_count=len(source_paths),
        curation_input_count=len(curation_rows),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        pending_count=len(pending) + len(curation_pending),
        training_ready=training_ready,
        class_counts=class_counts,
        independent_source_speaker_groups_by_class=groups_by_class,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        pending=(*podcast_pending, *[item for item in pending if item not in podcast_pending], *curation_pending),
        findings=tuple(findings),
    )
    return report, split_rows


def write_report(report: DatasetCurationReport, path: Path | str) -> None:
    destination = Path(path).expanduser()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing curation report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(report.as_dict(), file, indent=2)
        file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--curation-csv", type=Path, help="Optional manually curated evidence CSV; never generated by this tool")
    parser.add_argument("--report-output", type=Path, help="Optional new JSON curation report path")
    parser.add_argument("--manifest-output", type=Path, help="Optional new training manifest; only written when fully ready")
    args = parser.parse_args(argv)
    try:
        report, records = curate_dataset(args.project_root, curation_csv=args.curation_csv)
        print(report.format_text())
        if args.report_output is not None:
            write_report(report, args.report_output)
        if args.manifest_output is not None:
            if not report.training_ready:
                print("Training manifest not written: dataset lacks verified labels and/or sufficient independent groups.")
                return 1
            write_split_manifest(records, args.manifest_output)
        return 0 if report.training_ready else 1
    except (CurationError, DatasetReadinessError, FileExistsError, OSError, ValueError) as error:
        print(f"ERROR: dataset curation stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
