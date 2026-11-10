"""Audit labeled audio metadata and create leakage-resistant data partitions.

This module never assigns labels, downloads data, trains a model, or overwrites
an existing manifest. Split names intentionally match this repository's
trainer/evaluator conventions: ``train``, ``dev`` (validation), and ``eval``
(test).

Required metadata schema:
    file_path, label, split, source_id, source, speaker_id, utterance_id,
    language, permission_status, generation_method, label_source,
    label_provenance, attack_type

``source_id`` is the original recording/source unit; ``speaker_id`` must be a
stable pseudonymous speaker key. Both are kept in a split as indivisible
entities. For label 0, generation_method must be ``none`` or
``not_applicable``. For label 1, generation_method must identify the evidenced
synthesis method. ``label_source`` and ``label_provenance`` document the basis
for either label; filenames and ordinary podcast context are not label proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import soundfile as sf

METADATA_FIELDS = (
    "file_path",
    "label",
    "split",
    "source_id",
    "source",
    "speaker_id",
    "utterance_id",
    "language",
    "permission_status",
    "permission_evidence_ref",
    "generation_method",
    "label_source",
    "label_provenance",
    "attack_type",
)
SPLIT_NAMES = ("train", "dev", "eval")
SPLIT_SEMANTICS = {"train": "train", "dev": "validation", "eval": "test"}
CLASS_NAMES = {0: "bona_fide", 1: "spoof"}
VALID_AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".aiff", ".aif"}
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "eval": 0.15}
DEFAULT_IMBALANCE_WARNING_RATIO = 3.0


@dataclass(frozen=True)
class ReadinessFinding:
    severity: str
    code: str
    message: str
    row_number: int | None = None
    file_path: str = ""


@dataclass(frozen=True)
class DatasetReadinessReport:
    input_records: int
    output_records: int
    input_class_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    imbalance_ratio: float | None
    duplicate_audio_files: int
    unreadable_audio_files: int
    missing_audio_files: int
    source_leaks: int
    speaker_leaks: int
    findings: tuple[ReadinessFinding, ...]

    @property
    def error_count(self) -> int:
        return sum(item.severity == "ERROR" for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "WARNING" for item in self.findings)

    @property
    def ready(self) -> bool:
        return self.error_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_version": 1,
            "status": "ready" if self.ready else "not_ready",
            "input_records": self.input_records,
            "output_records": self.output_records,
            "input_class_counts": self.input_class_counts,
            "split_semantics": SPLIT_SEMANTICS,
            "class_counts": self.class_counts,
            "imbalance_ratio": self.imbalance_ratio,
            "duplicate_audio_files": self.duplicate_audio_files,
            "unreadable_audio_files": self.unreadable_audio_files,
            "missing_audio_files": self.missing_audio_files,
            "source_leaks": self.source_leaks,
            "speaker_leaks": self.speaker_leaks,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [item.__dict__ for item in self.findings],
            "interpretation": "Dataset readiness checks only; no model training or performance evaluation was performed.",
        }

    def format_text(self) -> str:
        lines = [
            "TRAINING DATA READINESS (no training performed)",
            "-" * 49,
            f"Input records: {self.input_records}",
            f"Validated records: {self.output_records}",
            f"Input class counts (0=bona_fide, 1=spoof): {self.input_class_counts}",
            f"Class counts by split (0=bona_fide, 1=spoof): {self.class_counts}",
            f"Global class imbalance ratio: {self.imbalance_ratio}",
            f"Duplicate audio files: {self.duplicate_audio_files}",
            f"Unreadable audio files: {self.unreadable_audio_files}",
            f"Missing audio files: {self.missing_audio_files}",
            f"Cross-split source leaks: {self.source_leaks}",
            f"Cross-split speaker leaks: {self.speaker_leaks}",
            f"Errors: {self.error_count}; warnings: {self.warning_count}",
        ]
        for finding in self.findings:
            location = f" row={finding.row_number}" if finding.row_number is not None else ""
            file_path = f" file={finding.file_path}" if finding.file_path else ""
            lines.append(f"[{finding.severity}] {finding.code}:{location}{file_path} {finding.message}")
        lines.append("Result: " + ("READY" if self.ready else "NOT READY"))
        return "\n".join(lines)


class DatasetReadinessError(ValueError):
    """Raised when input metadata cannot be safely interpreted or split."""


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _normalize_ratios(ratios: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    if set(values) != set(SPLIT_NAMES):
        raise DatasetReadinessError(f"Split ratios must define exactly: {', '.join(SPLIT_NAMES)}")
    if any(value <= 0 for value in values.values()):
        raise DatasetReadinessError("Every split ratio must be positive")
    total = sum(values.values())
    return {name: values[name] / total for name in SPLIT_NAMES}


def _validate_metadata_row(row: Mapping[str, str], row_number: int) -> list[ReadinessFinding]:
    findings: list[ReadinessFinding] = []
    for field in METADATA_FIELDS:
        if field not in row:
            findings.append(
                ReadinessFinding("ERROR", "missing_field", f"Required metadata field {field!r} is absent.", row_number)
            )
    if findings:
        return findings

    for field in METADATA_FIELDS:
        value = (row.get(field) or "").strip()
        if field in {"attack_type", "utterance_id"}:
            continue
        if field == "split":
            if value and value.casefold() not in {"unassigned", "pending"}:
                findings.append(
                    ReadinessFinding(
                        "ERROR",
                        "preassigned_split",
                        "Input split must be blank/unassigned; refusing to silently repartition existing data.",
                        row_number,
                    )
                )
            continue
        if (
            field == "generation_method"
            and row.get("label", "").strip() == "0"
            and value.casefold() in {"none", "not_applicable", "not applicable"}
        ):
            continue
        if field == "permission_evidence_ref" and value.casefold() in {
            "unknown",
            "unverified",
            "none",
            "n/a",
            "na",
        }:
            findings.append(
                ReadinessFinding(
                    "ERROR",
                    "missing_permission_evidence",
                    "Permission evidence must be explicitly referenced; status alone is not evidence.",
                    row_number,
                )
            )
            continue
        if field in {"label_source", "label_provenance"} and value.casefold() in {
            "unknown",
            "unverified",
            "none",
            "n/a",
            "na",
            "pending",
        }:
            findings.append(
                ReadinessFinding(
                    "ERROR",
                    "missing_label_evidence",
                    f"{field} must reference explicit label evidence, not a placeholder.",
                    row_number,
                )
            )
            continue
        if not value:
            findings.append(
                ReadinessFinding("ERROR", "missing_metadata", f"Required metadata value {field!r} is blank.", row_number)
            )

    raw_label = (row.get("label") or "").strip()
    try:
        label = int(raw_label)
    except ValueError:
        label = -1
    if label not in CLASS_NAMES:
        findings.append(
            ReadinessFinding("ERROR", "invalid_label", "label must be evidenced class 0 (bona fide) or 1 (spoof).", row_number)
        )
    elif label == 1 and (row.get("generation_method") or "").strip().casefold() in {
        "",
        "none",
        "not_applicable",
        "not applicable",
    }:
        findings.append(
            ReadinessFinding(
                "ERROR",
                "missing_generation_evidence",
                "Spoof label requires a documented synthesis/generation method; no label is inferred.",
                row_number,
            )
        )
    if (row.get("permission_status") or "").strip().casefold() != "granted":
        findings.append(
            ReadinessFinding(
                "ERROR",
                "permission_not_granted",
                "permission_status must be 'granted' for dataset use.",
                row_number,
            )
        )
    if (row.get("language") or "").strip().casefold() in {"", "unknown"}:
        findings.append(
            ReadinessFinding("WARNING", "unknown_language", "Language is unknown or blank.", row_number)
        )
    return findings


def read_labeled_metadata(path: Path | str) -> tuple[list[dict[str, str]], Path, list[ReadinessFinding]]:
    """Read an explicitly labeled metadata CSV without modifying it."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {input_path}")
    try:
        with input_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            missing_fields = sorted(set(METADATA_FIELDS) - set(headers))
            if missing_fields:
                raise DatasetReadinessError(
                    f"Metadata CSV missing required columns: {', '.join(missing_fields)}"
                )
            rows = [
                {field: (value or "").strip() for field, value in raw.items() if field is not None}
                for raw in reader
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise DatasetReadinessError(f"Could not read metadata CSV: {error}") from error

    findings: list[ReadinessFinding] = []
    for row_number, row in enumerate(rows, start=2):
        findings.extend(_validate_metadata_row(row, row_number))
        for field_name in ("permission_evidence_ref", "label_source", "label_provenance"):
            reference = (row.get(field_name) or "").strip()
            if not reference:
                continue
            if urlsplit(reference).scheme.casefold() in {"http", "https"}:
                findings.append(
                    ReadinessFinding(
                        "ERROR",
                        "external_evidence_not_locally_verified",
                        f"{field_name} must be locally available for review; URLs are not fetched.",
                        row_number,
                        reference,
                    )
                )
                continue
            evidence_path = Path(reference).expanduser()
            evidence_path = evidence_path.resolve() if evidence_path.is_absolute() else (input_path.parent / evidence_path).resolve()
            if not evidence_path.is_file():
                findings.append(
                    ReadinessFinding(
                        "ERROR",
                        "evidence_document_missing",
                        f"{field_name} does not resolve to an existing local file.",
                        row_number,
                        str(evidence_path),
                    )
                )
            elif evidence_path.stat().st_size == 0:
                findings.append(
                    ReadinessFinding(
                        "ERROR",
                        "evidence_document_empty",
                        f"{field_name} points to an empty local file.",
                        row_number,
                        str(evidence_path),
                    )
                )
    return rows, input_path.parent, findings


def _inspect_records(
    rows: Sequence[dict[str, str]],
    base_dir: Path,
    findings: list[ReadinessFinding],
) -> tuple[list[dict[str, str]], int, int, int]:
    hash_owners: dict[str, int] = {}
    path_owners: dict[Path, int] = {}
    duplicate_audio = unreadable_audio = missing_audio = 0
    validated: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        row_number = index + 2
        raw_path = row["file_path"].strip()
        path = Path(raw_path).expanduser()
        path = path.resolve() if path.is_absolute() else (base_dir / path).resolve()
        row["file_path"] = str(path)
        if path.suffix.casefold() not in VALID_AUDIO_SUFFIXES:
            findings.append(
                ReadinessFinding("ERROR", "unsupported_audio_format", "Unsupported or missing audio extension.", row_number, str(path))
            )
            continue
        if not path.is_file():
            missing_audio += 1
            findings.append(
                ReadinessFinding("ERROR", "missing_audio", "Audio file does not exist.", row_number, str(path))
            )
            continue
        previous_row = path_owners.get(path)
        if previous_row is not None:
            duplicate_audio += 1
            findings.append(
                ReadinessFinding(
                    "ERROR", "duplicate_audio_path", f"Same resolved audio path was already listed at row {previous_row}.", row_number, str(path)
                )
            )
            continue
        path_owners[path] = row_number
        try:
            digest = hashlib.sha256()
            with sf.SoundFile(path, mode="r") as audio:
                if audio.frames <= 0 or audio.samplerate <= 0 or audio.channels <= 0:
                    raise ValueError("Audio has no frames or invalid audio metadata")
                for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                    digest.update(block.tobytes())
            content_hash = digest.hexdigest()
        except (OSError, RuntimeError, sf.SoundFileRuntimeError, ValueError) as error:
            unreadable_audio += 1
            findings.append(
                ReadinessFinding("ERROR", "corrupt_audio", f"Could not fully decode audio ({type(error).__name__}).", row_number, str(path))
            )
            continue
        row["audio_sha256"] = content_hash
        previous_hash_row = hash_owners.get(content_hash)
        if previous_hash_row is not None:
            duplicate_audio += 1
            findings.append(
                ReadinessFinding(
                    "ERROR", "duplicate_audio_content", f"Audio content duplicates row {previous_hash_row}.", row_number, str(path)
                )
            )
            continue
        hash_owners[content_hash] = row_number
        validated.append(row)
    return validated, duplicate_audio, unreadable_audio, missing_audio


def _source_speaker_components(rows: Sequence[dict[str, str]]) -> list[list[int]]:
    union = _UnionFind()
    for index, row in enumerate(rows):
        source_node = f"source:{row['source_id']}"
        speaker_node = f"speaker:{row['speaker_id']}"
        record_node = f"record:{index}"
        union.union(record_node, source_node)
        union.union(record_node, speaker_node)
    components: dict[str, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(f"record:{index}")].append(index)
    return list(components.values())


def _partition_components(
    rows: Sequence[dict[str, str]],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[int, str]:
    components = _source_speaker_components(rows)
    if not components:
        raise DatasetReadinessError("No valid records available to split")
    class_component_count = {
        label: sum(any(int(rows[index]["label"]) == label for index in component) for component in components)
        for label in CLASS_NAMES
    }
    inadequate = [label for label, count in class_component_count.items() if count < len(SPLIT_NAMES)]
    if inadequate:
        missing_names = [CLASS_NAMES[label] for label in inadequate]
        raise DatasetReadinessError(
            "Cannot place every class in train/dev/eval without source or speaker leakage; "
            f"fewer than {len(SPLIT_NAMES)} independent source-speaker groups for {missing_names}."
        )

    totals = Counter(int(row["label"]) for row in rows)
    targets = {
        split: {label: totals[label] * ratios[split] for label in CLASS_NAMES}
        for split in SPLIT_NAMES
    }
    best_assignment: dict[int, str] | None = None
    best_score = float("inf")
    attempts = max(64, min(512, len(components) * 8))
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        ordered = list(components)
        rng.shuffle(ordered)
        ordered.sort(
            key=lambda component: (
                -len(component),
                -sum(int(rows[index]["label"]) == 0 for index in component),
                -sum(int(rows[index]["label"]) == 1 for index in component),
            )
        )
        split_counts = {split: Counter() for split in SPLIT_NAMES}
        assignment: dict[int, str] = {}
        split_tie_order = list(SPLIT_NAMES)
        rng.shuffle(split_tie_order)
        for component in ordered:
            component_counts = Counter(int(rows[index]["label"]) for index in component)
            scores: list[tuple[float, str]] = []
            for split in split_tie_order:
                score = 0.0
                for candidate_split in SPLIT_NAMES:
                    for label in CLASS_NAMES:
                        projected = split_counts[candidate_split][label]
                        if candidate_split == split:
                            projected += component_counts[label]
                        if projected == 0:
                            score += 100.0
                        target = targets[candidate_split][label]
                        score += ((projected - target) / max(target, 1.0)) ** 2
                scores.append((score, split))
            _, chosen = min(scores, key=lambda item: item[0])
            for index in component:
                assignment[index] = chosen
            split_counts[chosen].update(component_counts)

        absent_penalty = sum(
            1000.0
            for split in SPLIT_NAMES
            for label in CLASS_NAMES
            if split_counts[split][label] == 0
        )
        final_score = absent_penalty + sum(
            ((split_counts[split][label] - targets[split][label]) / max(targets[split][label], 1.0)) ** 2
            for split in SPLIT_NAMES
            for label in CLASS_NAMES
        )
        if final_score < best_score:
            best_score, best_assignment = final_score, assignment
        if absent_penalty == 0:
            break

    if best_assignment is None or any(
        not any(best_assignment[index] == split and int(row["label"]) == label for index, row in enumerate(rows))
        for split in SPLIT_NAMES
        for label in CLASS_NAMES
    ):
        raise DatasetReadinessError(
            "Could not assign every class to every split without source/speaker leakage; collect more independent groups."
        )
    return best_assignment


def validate_split_leakage(rows: Sequence[Mapping[str, str]]) -> list[ReadinessFinding]:
    """Return errors for source IDs or speakers represented across partitions."""
    source_splits: dict[str, set[str]] = defaultdict(set)
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    findings: list[ReadinessFinding] = []
    for row_number, row in enumerate(rows, start=2):
        split = str(row.get("split", "")).casefold()
        source_id = str(row.get("source_id", "")).strip()
        speaker_id = str(row.get("speaker_id", "")).strip()
        if source_id:
            source_splits[source_id].add(split)
        if speaker_id:
            speaker_splits[speaker_id].add(split)
    for source_id, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            findings.append(
                ReadinessFinding("ERROR", "source_leakage", f"Source appears in multiple splits: {sorted(splits)}.", file_path=source_id)
            )
    for speaker_id, splits in sorted(speaker_splits.items()):
        if len(splits) > 1:
            findings.append(
                ReadinessFinding("ERROR", "speaker_leakage", f"Speaker appears in multiple splits: {sorted(splits)}.", file_path=speaker_id)
            )
    return findings


def _distribution(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, dict[str, int]], float | None]:
    counters = {split: Counter() for split in SPLIT_NAMES}
    global_counts: Counter[int] = Counter()
    for row in rows:
        raw_label = str(row.get("label", "")).strip()
        if raw_label not in {str(label) for label in CLASS_NAMES}:
            continue
        label = int(raw_label)
        split = row["split"]
        global_counts[label] += 1
        if split in counters:
            counters[split][label] += 1
    distributions = {
        split: {
            "total": sum(counters[split].values()),
            "bona_fide": counters[split][0],
            "spoof": counters[split][1],
        }
        for split in SPLIT_NAMES
    }
    present = [count for count in global_counts.values() if count > 0]
    imbalance_ratio = max(present) / min(present) if len(present) == 2 else None
    return distributions, imbalance_ratio


def prepare_training_splits(
    input_csv: Path | str,
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
    imbalance_warning_ratio: float = DEFAULT_IMBALANCE_WARNING_RATIO,
) -> tuple[list[dict[str, str]], DatasetReadinessReport]:
    """Audit labeled records and assign source/speaker-disjoint train/dev/eval splits."""
    if seed < 0:
        raise DatasetReadinessError("seed must be non-negative")
    if imbalance_warning_ratio < 1:
        raise DatasetReadinessError("imbalance warning ratio must be at least 1")
    normalized_ratios = _normalize_ratios(ratios)
    rows, base_dir, findings = read_labeled_metadata(input_csv)
    validated, duplicate_audio, unreadable_audio, missing_audio = _inspect_records(rows, base_dir, findings)

    if not rows:
        findings.append(ReadinessFinding("ERROR", "empty_dataset", "No labeled metadata records were provided."))
    assigned: list[dict[str, str]] = []
    distributions, imbalance_ratio = _distribution(validated)
    has_both_classes = {
        int(row["label"])
        for row in validated
        if str(row.get("label", "")).strip() in {str(label) for label in CLASS_NAMES}
    } == set(CLASS_NAMES)
    if not has_both_classes:
        findings.append(
            ReadinessFinding(
                "ERROR",
                "missing_class",
                "Both bona-fide and spoof classes are required; labels are never inferred from audio or source type.",
            )
        )
    if not any(item.severity == "ERROR" for item in findings) and has_both_classes:
        try:
            assignment = _partition_components(validated, normalized_ratios, seed)
        except DatasetReadinessError as error:
            findings.append(ReadinessFinding("ERROR", "insufficient_independent_groups", str(error)))
        else:
            assigned = [dict(row, split=assignment[index]) for index, row in enumerate(validated)]
            findings.extend(validate_split_leakage(assigned))
            distributions, _assigned_imbalance_ratio = _distribution(assigned)
            for split in SPLIT_NAMES:
                for name in CLASS_NAMES.values():
                    if distributions[split][name] == 0:
                        findings.append(
                            ReadinessFinding("ERROR", "missing_class", f"{split} split has no {name} records.")
                        )

    if imbalance_ratio is not None and imbalance_ratio > imbalance_warning_ratio:
        findings.append(
            ReadinessFinding(
                "WARNING",
                "class_imbalance",
                f"Global majority/minority class ratio {imbalance_ratio:.2f} exceeds warning threshold {imbalance_warning_ratio:.2f}.",
            )
        )

    report = DatasetReadinessReport(
        input_records=len(rows),
        output_records=len(assigned),
        input_class_counts={
            CLASS_NAMES[label]: sum(1 for row in validated if row.get("label") == str(label))
            for label in CLASS_NAMES
        },
        class_counts=distributions,
        imbalance_ratio=imbalance_ratio,
        duplicate_audio_files=duplicate_audio,
        unreadable_audio_files=unreadable_audio,
        missing_audio_files=missing_audio,
        source_leaks=sum(item.code == "source_leakage" for item in findings),
        speaker_leaks=sum(item.code == "speaker_leakage" for item in findings),
        findings=tuple(findings),
    )
    return assigned, report


def audit_labeled_metadata(
    input_csv: Path | str,
    *,
    imbalance_warning_ratio: float = DEFAULT_IMBALANCE_WARNING_RATIO,
) -> DatasetReadinessReport:
    """Audit metadata/audio/class evidence without creating any partitions."""
    if imbalance_warning_ratio < 1:
        raise DatasetReadinessError("imbalance warning ratio must be at least 1")
    rows, base_dir, findings = read_labeled_metadata(input_csv)
    validated, duplicate_audio, unreadable_audio, missing_audio = _inspect_records(rows, base_dir, findings)
    distributions, imbalance_ratio = _distribution(validated)
    input_counts = {
        CLASS_NAMES[label]: sum(1 for row in validated if row.get("label") == str(label))
        for label in CLASS_NAMES
    }
    if not rows:
        findings.append(ReadinessFinding("ERROR", "empty_dataset", "No labeled metadata records were provided."))
    if any(count == 0 for count in input_counts.values()):
        missing = [name for name, count in input_counts.items() if count == 0]
        findings.append(
            ReadinessFinding(
                "ERROR",
                "missing_class",
                f"Both bona-fide and spoof classes are required; missing {missing}. Labels are never inferred from audio or source type.",
            )
        )
    if imbalance_ratio is not None and imbalance_ratio > imbalance_warning_ratio:
        findings.append(
            ReadinessFinding(
                "WARNING",
                "class_imbalance",
                f"Global majority/minority class ratio {imbalance_ratio:.2f} exceeds warning threshold {imbalance_warning_ratio:.2f}.",
            )
        )
    return DatasetReadinessReport(
        input_records=len(rows),
        output_records=len(validated),
        input_class_counts=input_counts,
        class_counts=distributions,
        imbalance_ratio=imbalance_ratio,
        duplicate_audio_files=duplicate_audio,
        unreadable_audio_files=unreadable_audio,
        missing_audio_files=missing_audio,
        source_leaks=0,
        speaker_leaks=0,
        findings=tuple(findings),
    )


def write_split_manifest(records: Sequence[Mapping[str, str]], output_path: Path | str) -> None:
    """Create a split manifest once; never overwrite existing files."""
    destination = Path(output_path).expanduser()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing split manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (*METADATA_FIELDS, "audio_sha256")
    output_file = destination.open("x", newline="", encoding="utf-8")
    try:
        with output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _parse_ratios(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("Split ratios must be JSON, e.g. {\"train\":0.7,\"dev\":0.15,\"eval\":0.15}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Split ratios must be a JSON object")
    try:
        return {str(key): float(amount) for key, amount in parsed.items()}
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Split ratios must contain numeric values") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Labeled audio metadata CSV using the documented schema")
    parser.add_argument("--output", type=Path, required=True, help="New output CSV; existing files are never overwritten")
    parser.add_argument("--report", type=Path, help="Optional JSON readiness report; existing files are never overwritten")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit source metadata/audio/classes only; do not generate or write split manifests.",
    )
    parser.add_argument("--ratios", type=_parse_ratios, default=DEFAULT_SPLIT_RATIOS)
    parser.add_argument("--imbalance-warning-ratio", type=float, default=DEFAULT_IMBALANCE_WARNING_RATIO)
    args = parser.parse_args(argv)
    try:
        if args.audit_only:
            report = audit_labeled_metadata(
                args.input,
                imbalance_warning_ratio=args.imbalance_warning_ratio,
            )
            print(report.format_text())
            return 0 if report.ready else 1
        records, report = prepare_training_splits(
            args.input,
            ratios=args.ratios,
            seed=args.seed,
            imbalance_warning_ratio=args.imbalance_warning_ratio,
        )
        print(report.format_text())
        if not report.ready:
            return 1
        if args.report is not None and args.report.exists():
            raise FileExistsError(f"Refusing to overwrite existing readiness report: {args.report}")
        write_split_manifest(records, args.output)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as file:
                json.dump(report.as_dict(), file, indent=2)
                file.write("\n")
        print(f"Split manifest created: {args.output}")
        if args.report is not None:
            print(f"Readiness report created: {args.report}")
        return 0
    except (OSError, DatasetReadinessError, ValueError) as error:
        print(f"ERROR: dataset readiness stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
