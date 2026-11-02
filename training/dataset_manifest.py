"""Build auditable manifests for ASVspoof 2019 LA and WaveFake.

This module never downloads datasets and never invents labels. ASVspoof labels
come only from protocol key columns. WaveFake's published generated-audio
archive is treated as an external spoof-only evaluation source, not as a
replacement for bona-fide training data or an official train/dev/eval split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

MANIFEST_FIELDS = (
    "file_path",
    "label",
    "split",
    "source",
    "attack_type",
    "speaker_id",
    "utterance_id",
)
ASVSPOOF_SOURCE = "ASVspoof2019_LA"
WAVEFAKE_SOURCE = "WaveFake"
SPLIT_PROTOCOL_CANDIDATES = {
    "train": ("train.trn.txt", "train.trn", "train.txt"),
    "dev": ("dev.trl.txt", "dev.trl", "dev.txt"),
    "eval": ("eval.trl.txt", "eval.trl", "eval.txt"),
}
AUDIO_SUFFIXES = {".flac", ".wav", ".wave"}
_KEY_TO_LABEL = {
    "bonafide": 0,
    "bona-fide": 0,
    "bona_fide": 0,
    "spoof": 1,
}


@dataclass(frozen=True)
class ManifestRecord:
    file_path: str
    label: int
    split: str
    source: str
    attack_type: str = ""
    speaker_id: str = ""
    utterance_id: str = ""


@dataclass(frozen=True)
class ProtocolRow:
    speaker_id: str
    utterance_id: str
    label: int
    attack_type: str


class ManifestError(ValueError):
    """Raised when dataset metadata cannot be safely interpreted."""


def parse_protocol_file(path: Path, split: str) -> list[ProtocolRow]:
    """Parse whitespace-separated ASVspoof CM protocol records.

    Expected rows have speaker ID, utterance ID, optional metadata columns, and
    a bona-fide/spoof key. Protocol key text is the sole label authority.
    """
    rows: list[ProtocolRow] = []
    with path.open("r", encoding="utf-8-sig") as protocol:
        for line_number, line in enumerate(protocol, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 3:
                raise ManifestError(
                    f"{path}:{line_number}: expected speaker, utterance, and key columns"
                )

            key = fields[-1].casefold()
            if key not in _KEY_TO_LABEL:
                raise ManifestError(
                    f"{path}:{line_number}: unsupported/missing protocol key {fields[-1]!r}; "
                    "expected bona-fide or spoof"
                )

            # ASVspoof 2019 LA protocols place the attack ID in an optional
            # metadata column; '-' denotes bona-fide/no attack. Retain an
            # available non-placeholder value without guessing its meaning.
            attack_type = ""
            if _KEY_TO_LABEL[key] == 1:
                candidates = [
                    token
                    for token in fields[2:-1]
                    if token not in {"-", "--", "?"}
                ]
                if candidates:
                    attack_type = candidates[-1]

            rows.append(
                ProtocolRow(
                    speaker_id=fields[0],
                    utterance_id=Path(fields[1]).stem,
                    label=_KEY_TO_LABEL[key],
                    attack_type=attack_type,
                )
            )

    if not rows:
        raise ManifestError(f"No protocol rows found in {path} for split {split!r}")
    return rows


def _find_protocol(protocol_dir: Path, split: str) -> Path:
    """Find the named official LA protocol file without guessing alternatives."""
    candidates = SPLIT_PROTOCOL_CANDIDATES[split]
    matches = [
        candidate
        for candidate in protocol_dir.rglob("ASVspoof2019.LA.cm.*")
        if candidate.is_file() and any(candidate.name.endswith(suffix) for suffix in candidates)
    ]
    if len(matches) != 1:
        found = ", ".join(str(item) for item in matches) or "none"
        raise ManifestError(
            f"Expected exactly one {split} LA CM protocol under {protocol_dir}; found: {found}"
        )
    return matches[0]


def _audio_index(audio_root: Path) -> dict[str, list[Path]]:
    if not audio_root.is_dir():
        raise ManifestError(f"Audio root does not exist or is not a directory: {audio_root}")
    index: dict[str, list[Path]] = defaultdict(list)
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES:
            index[path.stem].append(path.resolve())
    return index


def _split_match(path: Path, split: str) -> bool:
    aliases = {"dev": {"dev", "development"}, "train": {"train"}, "eval": {"eval", "evaluation"}}
    return any(part.casefold() in aliases[split] for part in path.parts)


def _resolve_protocol_audio(
    utterance_id: str,
    split: str,
    index: dict[str, list[Path]],
) -> Path | None:
    candidates = index.get(utterance_id, [])
    matching_partition = [path for path in candidates if _split_match(path, split)]
    if len(matching_partition) == 1:
        return matching_partition[0]
    if not matching_partition and len(candidates) == 1:
        return candidates[0]
    return None


def build_asvspoof_records(protocol_dir: Path, audio_root: Path) -> list[ManifestRecord]:
    """Build train/dev/eval records from official protocols and extracted audio."""
    protocols = {split: _find_protocol(protocol_dir, split) for split in ("train", "dev", "eval")}
    audio_by_stem = _audio_index(audio_root)
    records: list[ManifestRecord] = []
    for split, protocol_path in protocols.items():
        for row in parse_protocol_file(protocol_path, split):
            audio_path = _resolve_protocol_audio(row.utterance_id, split, audio_by_stem)
            records.append(
                ManifestRecord(
                    file_path=str(audio_path) if audio_path else "",
                    label=row.label,
                    split=split,
                    source=ASVSPOOF_SOURCE,
                    attack_type=row.attack_type,
                    speaker_id=row.speaker_id,
                    utterance_id=row.utterance_id,
                )
            )
    return records


def _wavefake_attack_type(path: Path, root: Path) -> str:
    """Use an actual relative directory name as generator metadata if present."""
    relative_parts = path.relative_to(root).parts[:-1]
    ignored = {"generated_audio", "generated", "audio", "wavs", "wav"}
    useful = [part for part in relative_parts if part.casefold() not in ignored]
    return useful[-1] if useful else ""


def build_wavefake_records(audio_root: Path) -> list[ManifestRecord]:
    """Index the official generated-audio archive as external spoof evaluation.

    WaveFake's published archive contains generated clips; its source corpora
    are not redistributed in that record. Therefore this function emits only
    label-1 records in a single `external_eval` split and makes no bona-fide
    records or invented train/dev partitions.
    """
    if not audio_root.is_dir():
        raise ManifestError(f"WaveFake root does not exist or is not a directory: {audio_root}")
    paths = sorted(
        (
            path.resolve()
            for path in audio_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )
    if not paths:
        raise ManifestError(f"No supported audio files found under {audio_root}")
    return [
        ManifestRecord(
            file_path=str(path),
            label=1,
            split="external_eval",
            source=WAVEFAKE_SOURCE,
            attack_type=_wavefake_attack_type(path, audio_root.resolve()),
            utterance_id=path.stem,
        )
        for path in paths
    ]


def write_manifest(records: Iterable[ManifestRecord], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
            count += 1
    return count


def _inspect_audio(path: Path) -> tuple[int, int, int]:
    """Read/decode every block; return sample rate, channels, and frame count."""
    with sf.SoundFile(path, mode="r") as audio:
        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        frame_count = int(audio.frames)
        for _block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            pass
    return sample_rate, channels, frame_count


def create_quality_report(
    records: Sequence[ManifestRecord], expected_sample_rate: int = 16000
) -> dict[str, object]:
    """Validate references/audio and summarize distributions and leakage signals."""
    counts_by_split: dict[str, Counter[int]] = defaultdict(Counter)
    source_speaker_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    utterance_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    attack_type_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    recording_path_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    durations_seconds: list[float] = []
    missing_paths = unreadable_paths = sample_rate_mismatches = zero_frame_files = 0
    empty_path_records = duplicate_keys = 0
    record_keys: set[tuple[str, str, str]] = set()
    issue_examples: list[dict[str, str]] = []

    def note_issue(kind: str, record: ManifestRecord, detail: str) -> None:
        if len(issue_examples) < 50:
            issue_examples.append(
                {"kind": kind, "utterance_id": record.utterance_id, "detail": detail}
            )

    for record in records:
        counts_by_split[record.split][record.label] += 1
        if record.speaker_id:
            source_speaker_splits[(record.source, record.speaker_id)].add(record.split)
        if record.utterance_id:
            utterance_splits[(record.source, record.utterance_id)].add(record.split)
        if record.attack_type:
            attack_type_splits[(record.source, record.attack_type)].add(record.split)
        if record.file_path:
            recording_path_splits[(record.source, str(Path(record.file_path).resolve()))].add(
                record.split
            )

        key = (record.source, record.split, record.utterance_id)
        if key in record_keys:
            duplicate_keys += 1
            note_issue("duplicate_record", record, "Same source/split/utterance ID occurs more than once")
        record_keys.add(key)

        if not record.file_path:
            empty_path_records += 1
            missing_paths += 1
            note_issue("missing_audio", record, "No unique audio file matched this protocol utterance")
            continue

        audio_path = Path(record.file_path)
        if not audio_path.is_file():
            missing_paths += 1
            note_issue("missing_audio", record, "Manifest path does not exist or is not a file")
            continue
        try:
            sample_rate, channel_count, frame_count = _inspect_audio(audio_path)
            sample_rates[sample_rate] += 1
            channels[channel_count] += 1
            if frame_count <= 0:
                zero_frame_files += 1
                note_issue("empty_audio", record, "Audio contains zero frames")
            durations_seconds.append(frame_count / sample_rate if sample_rate else 0.0)
            if sample_rate != expected_sample_rate:
                sample_rate_mismatches += 1
                note_issue(
                    "sample_rate_mismatch",
                    record,
                    f"Expected {expected_sample_rate} Hz; found {sample_rate} Hz",
                )
        except (OSError, RuntimeError, sf.SoundFileRuntimeError, ValueError) as exc:
            unreadable_paths += 1
            note_issue("unreadable_audio", record, f"{type(exc).__name__}: {exc}")

    speaker_leaks = [
        {"source": source, "speaker_id": speaker, "splits": sorted(splits)}
        for (source, speaker), splits in sorted(source_speaker_splits.items())
        if len(splits) > 1
    ]
    utterance_leaks = [
        {"source": source, "utterance_id": utterance, "splits": sorted(splits)}
        for (source, utterance), splits in sorted(utterance_splits.items())
        if len(splits) > 1
    ]
    attack_type_overlaps = [
        {"source": source, "attack_type": attack_type, "splits": sorted(splits)}
        for (source, attack_type), splits in sorted(attack_type_splits.items())
        if len(splits) > 1
    ]
    evaluation_attack_overlaps = [
        entry
        for entry in attack_type_overlaps
        if "eval" in entry["splits"]
        and any(split in {"train", "dev"} for split in entry["splits"])
    ]
    recording_path_overlaps = [
        {"source": source, "file_path": file_path, "splits": sorted(splits)}
        for (source, file_path), splits in sorted(recording_path_splits.items())
        if len(splits) > 1
    ]
    distributions = {
        split: {
            "total": sum(counter.values()),
            "bona_fide_label_0": counter.get(0, 0),
            "spoof_label_1": counter.get(1, 0),
        }
        for split, counter in sorted(counts_by_split.items())
    }
    issue_count = (
        missing_paths
        + unreadable_paths
        + sample_rate_mismatches
        + zero_frame_files
        + duplicate_keys
        + len(speaker_leaks)
        + len(utterance_leaks)
        + len(recording_path_overlaps)
        + len(evaluation_attack_overlaps)
    )
    return {
        "report_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "record_count": len(records),
        "label_mapping": {"bona_fide": 0, "spoof": 1},
        "expected_sample_rate_hz": expected_sample_rate,
        "class_distribution_by_split": distributions,
        "audio_validation": {
            "missing_paths": missing_paths,
            "empty_path_records": empty_path_records,
            "unreadable_paths": unreadable_paths,
            "zero_frame_files": zero_frame_files,
            "sample_rate_mismatches": sample_rate_mismatches,
            "observed_sample_rate_counts": dict(sorted(sample_rates.items())),
            "observed_channel_counts": dict(sorted(channels.items())),
            "duration_seconds": {
                "total": round(sum(durations_seconds), 3),
                "minimum": round(min(durations_seconds), 3) if durations_seconds else None,
                "maximum": round(max(durations_seconds), 3) if durations_seconds else None,
            },
        },
        "leakage_checks": {
            "cross_split_speaker_groups": len(speaker_leaks),
            "cross_split_utterance_ids": len(utterance_leaks),
            "cross_split_attack_type_groups": len(attack_type_overlaps),
            "evaluation_attack_types_seen_in_train_or_dev": len(evaluation_attack_overlaps),
            "evaluation_attack_overlap_examples": evaluation_attack_overlaps[:50],
            "same_recording_path_in_multiple_splits": len(recording_path_overlaps),
            "speaker_overlap_examples": speaker_leaks[:50],
            "utterance_overlap_examples": utterance_leaks[:50],
            "attack_type_overlap_examples": attack_type_overlaps[:50],
            "recording_path_overlap_examples": recording_path_overlaps[:50],
            "note": (
                "Speaker, attack-system, utterance, and exact-path overlap checks use only available metadata. "
                "Train/dev attack overlap can be part of the official protocol; evaluation attack overlap "
                "with train/dev is reported as a leakage concern. Content-identical files under different "
                "paths are not detected."
            ),
        },
        "duplicate_source_split_utterance_records": duplicate_keys,
        "issue_count": issue_count,
        "issue_examples_capped_at_50": issue_examples,
        "interpretation": "Quality checks only; this report is not a model evaluation or detection result.",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ASVspoof 2019 LA or WaveFake audio manifest and quality report."
    )
    parser.add_argument(
        "--dataset",
        choices=("asvspoof2019-la", "wavefake"),
        required=True,
        help="Dataset format to index; files must already be obtained from official sources.",
    )
    parser.add_argument("--audio-root", type=Path, required=True, help="Extracted audio directory")
    parser.add_argument("--output", type=Path, help="CSV path (default under data/manifests)")
    parser.add_argument("--report", type=Path, help="JSON report path (default under reports)")
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        help="ASVspoof protocol directory containing official LA CM train/dev/eval files",
    )
    parser.add_argument(
        "--expected-sample-rate",
        type=int,
        default=16000,
        help="Expected sample rate in Hz (default: 16000); validation does not resample files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if quality checks detect missing/invalid audio or leakage",
    )
    args = parser.parse_args(argv)
    if args.expected_sample_rate <= 0:
        parser.error("--expected-sample-rate must be positive")
    if args.dataset == "asvspoof2019-la" and args.protocol_dir is None:
        parser.error("--protocol-dir is required for ASVspoof 2019 LA")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = args.output or Path("data/manifests") / f"{args.dataset}.csv"
    report_path = args.report or Path("reports") / f"{args.dataset}_data_quality.json"

    try:
        if args.dataset == "asvspoof2019-la":
            records = build_asvspoof_records(args.protocol_dir, args.audio_root)
        else:
            records = build_wavefake_records(args.audio_root)
    except (ManifestError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    count = write_manifest(records, output_path)
    report = create_quality_report(records, args.expected_sample_rate)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Manifest: {output_path} ({count} records)")
    print(f"Quality report: {report_path}")
    print(json.dumps(report["class_distribution_by_split"], indent=2))
    print(
        "Audio issues: "
        f"missing={report['audio_validation']['missing_paths']}, "
        f"unreadable={report['audio_validation']['unreadable_paths']}, "
        f"sample-rate-mismatches={report['audio_validation']['sample_rate_mismatches']}; "
        "cross-split speakers="
        f"{report['leakage_checks']['cross_split_speaker_groups']}"
    )
    print("No model training was performed.")

    if args.strict and report["issue_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
