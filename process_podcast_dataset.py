"""Safe, explicit-input pipeline for local podcast media processing.

This pipeline never downloads media. It preserves source files and only writes
outputs that do not already exist. Diarization is performed on each complete
recording before clip segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf

from preflight_dataset import (
    MANIFEST_FIELDS,
    SOURCE_ID_PATTERN,
    SUPPORTED_MEDIA_SUFFIXES,
)

CLIP_METADATA_FIELDS = (
    "source_id",
    "clip_id",
    "file_path",
    "clip_start_seconds",
    "clip_end_seconds",
    "duration",
    "sample_rate",
    "speaker_ids",
    "speaker_turns",
    "quality_status",
    "quality_issues",
)
LEGACY_CLIP_METADATA_FIELDS = (
    "clip_id",
    "source_id",
    "file_path",
    "duration",
    "sample_rate",
    "language",
    "speaker_id",
    "quality_status",
)
SPEAKER_TURN_FIELDS = ("source_id", "start_seconds", "end_seconds", "speaker_id")
SEGMENT_SECONDS = 10
TARGET_SAMPLE_RATE = 16000


class PipelineError(RuntimeError):
    """Raised when input validation or processing fails safely."""


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str


@dataclass(frozen=True)
class ClipMetadata:
    source_id: str
    clip_id: str
    file_path: str
    clip_start_seconds: float
    clip_end_seconds: float
    duration: float
    sample_rate: int
    speaker_ids: str
    speaker_turns: str
    quality_status: str
    quality_issues: str

    def as_row(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "clip_id": self.clip_id,
            "file_path": self.file_path,
            "clip_start_seconds": f"{self.clip_start_seconds:.3f}",
            "clip_end_seconds": f"{self.clip_end_seconds:.3f}",
            "duration": f"{self.duration:.6f}",
            "sample_rate": str(self.sample_rate),
            "speaker_ids": self.speaker_ids,
            "speaker_turns": self.speaker_turns,
            "quality_status": self.quality_status,
            "quality_issues": self.quality_issues,
        }


@dataclass(frozen=True)
class SourceResult:
    source_id: str
    status: str
    message: str
    clips: int = 0


@dataclass(frozen=True)
class PipelineResult:
    sources: tuple[SourceResult, ...]

    @property
    def error_count(self) -> int:
        return sum(item.status == "ERROR" for item in self.sources)

    @property
    def waiting_count(self) -> int:
        return sum(item.status == "WAITING" for item in self.sources)

    @property
    def processed_count(self) -> int:
        return sum(item.status == "PROCESSED" for item in self.sources)

    @property
    def skipped_count(self) -> int:
        return sum(item.status == "SKIPPED" for item in self.sources)

    def format_text(self) -> str:
        lines = ["LOCAL AUDIO PROCESSING", "-" * 28]
        if not self.sources:
            lines.append("No source records found.")
        for item in self.sources:
            lines.append(f"[{item.status}] {item.source_id}: {item.message}")
        lines.extend(
            (
                "",
                (
                    f"Processed: {self.processed_count}; skipped: {self.skipped_count}; "
                    f"waiting for local media: {self.waiting_count}; errors: {self.error_count}"
                ),
            )
        )
        return "\n".join(lines)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PipelineError(f"Processing manifest not found: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            missing = sorted(set(MANIFEST_FIELDS) - set(headers))
            if missing:
                raise PipelineError(f"Processing manifest missing columns: {', '.join(missing)}")
            rows = [
                {field: (row.get(field) or "").strip() for field in headers}
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise PipelineError(f"Could not read processing manifest: {error}") from error

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        source_id = row.get("source_id", "")
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise PipelineError(f"Invalid or missing source_id at manifest row {row_number}: {source_id!r}")
        if source_id in seen:
            raise PipelineError(f"Duplicate source_id in processing manifest: {source_id}")
        seen.add(source_id)
        if not row.get("permission_status"):
            raise PipelineError(f"Missing permission_status for {source_id}")
    return rows


def _parse_media_arguments(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PipelineError(f"--media must use SOURCE_ID=LOCAL_PATH, got {value!r}")
        source_id, path = value.split("=", 1)
        source_id, path = source_id.strip(), path.strip()
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise PipelineError(f"Invalid source ID in --media argument: {source_id!r}")
        if not path:
            raise PipelineError(f"Empty local media path for {source_id}")
        if source_id in result:
            raise PipelineError(f"Multiple local media paths supplied for {source_id}")
        result[source_id] = path
    return result


def _local_path(raw_path: str, root: Path, source_id: str, description: str) -> Path:
    if urlsplit(raw_path).scheme.casefold() in {"http", "https"}:
        raise PipelineError(f"{description} for {source_id} is a URL, not a local file; downloads are disabled")
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PipelineError(f"{description} for {source_id} must be inside the project root") from error
    return resolved


def _validate_media(
    source_id: str,
    raw_path: str,
    root: Path,
    permission_status: str,
) -> Path:
    if permission_status.casefold() != "granted":
        raise PipelineError(f"Permission status for {source_id} is {permission_status!r}; required status is 'granted'")
    media_path = _local_path(raw_path, root, source_id, "Media path")
    if media_path.suffix.casefold() not in SUPPORTED_MEDIA_SUFFIXES:
        raise PipelineError(f"Unsupported media extension for {source_id}: {media_path.suffix or '<none>'}")
    if not media_path.is_file():
        raise FileNotFoundError(f"Local media file is not present for {source_id}: {media_path}")
    return media_path


def _output_paths(row: dict[str, str], root: Path, source_id: str) -> tuple[Path, Path, Path]:
    dataset_dir = root / "PodcastDataset"
    mp3_value = row.get("mp3_file", "")
    clips_value = row.get("wav_folder", "")
    if not mp3_value or not clips_value:
        raise PipelineError(f"Manifest must provide mp3_file and wav_folder for {source_id}")
    mp3_path = _local_path(mp3_value, root, source_id, "MP3 output")
    clips_dir = _local_path(clips_value, root, source_id, "Clip output directory")
    if mp3_path.suffix.casefold() != ".mp3":
        raise PipelineError(f"mp3_file output for {source_id} must end in .mp3")
    turns_path = dataset_dir / "diarization" / f"{source_id}_speakers.csv"
    _local_path(str(turns_path), root, source_id, "Speaker-turn output")
    return mp3_path, turns_path, clips_dir


def _ensure_unique_outputs(rows: Sequence[dict[str, str]], root: Path) -> None:
    owners: dict[Path, tuple[str, bool]] = {}
    for row in rows:
        source_id = row["source_id"]
        mp3_path, turns_path, clips_dir = _output_paths(row, root, source_id)
        for path, is_directory in ((mp3_path, False), (turns_path, False), (clips_dir, True)):
            for existing_path, (existing_owner, existing_is_directory) in owners.items():
                overlaps = path == existing_path or (
                    is_directory and existing_path.is_relative_to(path)
                ) or (existing_is_directory and path.is_relative_to(existing_path))
                if not overlaps:
                    continue
                raise PipelineError(
                    f"Duplicate output path (or overlap) for {source_id} and {existing_owner}: {path}"
                )
            owners[path] = (source_id, is_directory)


def _read_existing_csv(path: Path, fieldnames: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            current_fields = reader.fieldnames or []
            if current_fields != list(fieldnames):
                if current_fields == list(LEGACY_CLIP_METADATA_FIELDS) and not list(reader):
                    # The repository currently has only this empty legacy header.
                    # No data is discarded; the schema advances on first success.
                    return []
                raise PipelineError(
                    f"Existing CSV has an unexpected schema and will not be overwritten: {path}"
                )
            return [{field: (row.get(field) or "") for field in fieldnames} for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise PipelineError(f"Could not read existing CSV {path}: {error}") from error


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _audio_quality(path: Path) -> tuple[float, int, str, str]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        return 0.0, 0, "INVALID", f"unreadable_audio:{type(error).__name__}"
    if audio.size == 0 or sample_rate <= 0:
        return 0.0, sample_rate, "INVALID", "empty_audio"

    duration = audio.shape[0] / sample_rate
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    issues: list[str] = []
    if sample_rate != TARGET_SAMPLE_RATE:
        issues.append("incorrect_sample_rate")
    if audio.shape[1] != 1:
        issues.append("not_mono")
    if duration < 0.1:
        issues.append("clip_too_short")
    if rms < 0.001:
        issues.append("silent_or_very_quiet")
    if peak >= 0.999:
        issues.append("possible_clipping")
    return duration, sample_rate, "CHECK" if issues else "PASS", ";".join(issues)


def _write_speaker_turns(path: Path, source_id: str, turns: Sequence[SpeakerTurn]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SPEAKER_TURN_FIELDS)
        writer.writeheader()
        for turn in turns:
            writer.writerow(
                {
                    "source_id": source_id,
                    "start_seconds": f"{turn.start:.3f}",
                    "end_seconds": f"{turn.end:.3f}",
                    "speaker_id": turn.speaker_id,
                }
            )


def _parse_diarization_result(result: object) -> list[SpeakerTurn]:
    diarization = getattr(result, "speaker_diarization", result)
    if not hasattr(diarization, "itertracks"):
        raise PipelineError("Diarization output does not provide speaker turns")
    turns: list[SpeakerTurn] = []
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        start, end = float(segment.start), float(segment.end)
        if start < 0 or end <= start:
            continue
        turns.append(SpeakerTurn(start, end, str(speaker)))
    return sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker_id))


def _segment_recording(
    ffmpeg: str,
    mp3_path: Path,
    clips_stage: Path,
) -> list[Path]:
    clips_stage.mkdir(parents=True, exist_ok=True)
    output_pattern = clips_stage / "clip_%04d.wav"
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_SECONDS),
        "-reset_timestamps",
        "1",
        "-n",
        str(output_pattern),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    clips = sorted(clips_stage.glob("clip_*.wav"))
    if not clips:
        raise PipelineError(f"FFmpeg produced no WAV clips from {mp3_path.name}")
    return clips


def _process_one(
    row: dict[str, str],
    media_path: Path,
    root: Path,
    stage_root: Path,
    ffmpeg: str,
    diarize: Callable[[Path], Iterable[SpeakerTurn]],
) -> tuple[list[ClipMetadata], Path, Path, Path, Path, Path]:
    source_id = row["source_id"]
    mp3_path, _turns_path, clips_dir = _output_paths(row, root, source_id)
    source_stage = stage_root / source_id
    source_stage.mkdir(parents=True)
    staged_mp3 = source_stage / "recording.mp3"
    staged_turns = source_stage / "speaker_turns.csv"
    staged_clips = source_stage / "clips"

    convert_command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        "-n",
        str(staged_mp3),
    ]
    subprocess.run(convert_command, check=True, capture_output=True, text=True)
    if not staged_mp3.is_file() or staged_mp3.stat().st_size == 0:
        raise PipelineError(f"FFmpeg did not create an MP3 for {source_id}")

    turns = sorted(diarize(staged_mp3), key=lambda turn: (turn.start, turn.end, turn.speaker_id))
    _write_speaker_turns(staged_turns, source_id, turns)
    clips = _segment_recording(ffmpeg, staged_mp3, staged_clips)

    metadata: list[ClipMetadata] = []
    for index, clip_path in enumerate(clips):
        clip_start = index * SEGMENT_SECONDS
        duration, sample_rate, quality_status, quality_issues = _audio_quality(clip_path)
        if quality_status == "INVALID":
            raise PipelineError(f"Audio-quality validation could not read generated clip {clip_path.name}")
        clip_end = clip_start + duration
        intersecting = [
            {
                "speaker_id": turn.speaker_id,
                "start_seconds": round(max(turn.start, clip_start), 3),
                "end_seconds": round(min(turn.end, clip_end), 3),
            }
            for turn in turns
            if max(turn.start, clip_start) < min(turn.end, clip_end)
        ]
        speakers = sorted({item["speaker_id"] for item in intersecting})
        clip_id = f"{source_id}_{index:04d}"
        relative_clip_path = (clips_dir / clip_path.name).relative_to(root).as_posix()
        metadata.append(
            ClipMetadata(
                source_id=source_id,
                clip_id=clip_id,
                file_path=relative_clip_path,
                clip_start_seconds=float(clip_start),
                clip_end_seconds=float(clip_end),
                duration=duration,
                sample_rate=sample_rate,
                speaker_ids=";".join(speakers),
                speaker_turns=json.dumps(intersecting, separators=(",", ":")),
                quality_status=quality_status,
                quality_issues=quality_issues,
            )
        )
    return metadata, staged_mp3, staged_turns, staged_clips, mp3_path, clips_dir


def _write_rows_to_temp(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]]) -> Path:
    return _atomic_write_csv(path, fields, rows)


def resolve_ffmpeg(
    configured_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Find and verify FFmpeg using explicit setting, FFMPEG_PATH, then PATH."""
    environment = os.environ if environ is None else environ
    explicit_value = str(configured_path).strip() if configured_path is not None else ""
    environment_value = environment.get("FFMPEG_PATH", "").strip()
    setting = explicit_value or environment_value

    if setting:
        candidate_path = Path(setting).expanduser()
        resolved = str(candidate_path.resolve()) if candidate_path.is_file() else shutil.which(setting)
        if not resolved:
            source = "--ffmpeg" if explicit_value else "FFMPEG_PATH"
            raise PipelineError(
                f"FFmpeg configured by {source} was not found. Set it to an executable file or add FFmpeg to PATH."
            )
    else:
        resolved = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not resolved:
            raise PipelineError(
                "FFmpeg is unavailable. Add ffmpeg to PATH, set FFMPEG_PATH, or pass --ffmpeg with its executable path."
            )

    try:
        check = subprocess.run(
            [resolved, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PipelineError(
            f"FFmpeg executable could not be started ({type(error).__name__}). Check PATH or the configured executable."
        ) from error
    if check.returncode != 0 or "ffmpeg version" not in (check.stdout + check.stderr).casefold():
        raise PipelineError("Configured FFmpeg executable failed its -version check")
    return resolved


def _validate_processed_state(
    row: dict[str, str], root: Path, metadata_rows: Sequence[dict[str, str]]
) -> None:
    source_id = row["source_id"]
    mp3_path, turns_path, clips_dir = _output_paths(row, root, source_id)
    source_clips = [clip for clip in clips_dir.glob("*.wav")] if clips_dir.is_dir() else []
    metadata_for_source = [item for item in metadata_rows if item.get("source_id") == source_id]
    if not mp3_path.is_file() or not turns_path.is_file() or not source_clips or not metadata_for_source:
        raise PipelineError(
            f"{source_id} is marked processed but outputs/metadata are incomplete; refusing to overwrite"
        )


def run_pipeline(
    project_root: Path,
    media_overrides: dict[str, str] | None = None,
    *,
    ffmpeg: str | Path | None = None,
    diarizer: Callable[[Path], Iterable[SpeakerTurn]] | None = None,
    diarization_model: str = "pyannote/speaker-diarization-3.1",
) -> PipelineResult:
    """Run eligible manifest rows; missing local media is a non-error waiting state."""
    root = project_root.resolve()
    dataset_dir = root / "PodcastDataset"
    manifest_path = dataset_dir / "processing_manifest.csv"
    metadata_path = dataset_dir / "clip_metadata.csv"
    media_overrides = dict(media_overrides or {})
    rows = _read_manifest(manifest_path)
    source_ids = {row["source_id"] for row in rows}
    unknown_overrides = sorted(set(media_overrides) - source_ids)
    if unknown_overrides:
        raise PipelineError(f"Media supplied for unknown source ID(s): {', '.join(unknown_overrides)}")
    _ensure_unique_outputs(rows, root)

    metadata_rows = _read_existing_csv(metadata_path, CLIP_METADATA_FIELDS)
    metadata_ids = [item.get("clip_id", "") for item in metadata_rows]
    if len(metadata_ids) != len(set(metadata_ids)):
        raise PipelineError("Existing clip_metadata.csv contains duplicate clip_id values")

    results: list[SourceResult] = []
    candidates: list[tuple[dict[str, str], Path]] = []
    for row in rows:
        source_id = row["source_id"]
        status = row.get("processing_status", "").casefold()
        mp3_path, source_turns_path, clips_dir = _output_paths(row, root, source_id)
        if row.get("permission_status", "").casefold() != "granted":
            results.append(
                SourceResult(
                    source_id,
                    "ERROR",
                    f"permission_status must be 'granted'; found {row.get('permission_status') or '<blank>'!r}",
                )
            )
            continue
        if status == "processed":
            _validate_processed_state(row, root, metadata_rows)
            results.append(SourceResult(source_id, "SKIPPED", "already processed; outputs preserved"))
            continue
        if status not in {"waiting_for_audio", "pending", "", "ready"}:
            results.append(SourceResult(source_id, "ERROR", f"unsupported processing_status {status!r}"))
            continue

        raw_media = media_overrides.get(source_id)
        if raw_media is None:
            stored_path = row.get("original_video", "")
            if stored_path:
                stored_resolved = _local_path(stored_path, root, source_id, "Media path")
                if stored_resolved.is_file():
                    raw_media = stored_path
        if raw_media is None:
            results.append(SourceResult(source_id, "WAITING", "waiting for an explicitly supplied local media file"))
            continue
        try:
            media_path = _validate_media(source_id, raw_media, root, row["permission_status"])
            if media_path in {mp3_path.resolve(), source_turns_path.resolve()} or media_path == clips_dir.resolve():
                raise PipelineError(f"Input media for {source_id} conflicts with an output path")
            for output in (mp3_path, source_turns_path, clips_dir):
                if output.exists():
                    raise PipelineError(f"Output already exists; refusing to overwrite: {output}")
            if any(item.get("source_id") == source_id for item in metadata_rows):
                raise PipelineError(f"Metadata already exists for pending source {source_id}; refusing to overwrite")
            candidates.append((row, media_path))
        except FileNotFoundError as error:
            results.append(SourceResult(source_id, "WAITING", str(error)))
        except PipelineError as error:
            results.append(SourceResult(source_id, "ERROR", str(error)))

    if not candidates:
        return PipelineResult(tuple(results))

    # Detect duplicate source input paths before any processing starts.
    inputs: dict[Path, list[str]] = defaultdict(list)
    for row, media_path in candidates:
        inputs[media_path].append(row["source_id"])
    duplicate_inputs = {path for path, owners in inputs.items() if len(owners) > 1}
    for path in duplicate_inputs:
        owners = ", ".join(sorted(inputs[path]))
        for source_id in inputs[path]:
            results.append(
                SourceResult(source_id, "ERROR", f"Input media path is assigned to multiple sources: {owners} ({path})")
            )
    candidates = [(row, path) for row, path in candidates if path not in duplicate_inputs]

    if not candidates:
        return PipelineResult(tuple(results))

    try:
        resolved_ffmpeg = resolve_ffmpeg(ffmpeg)
    except PipelineError as error:
        for row, _ in candidates:
            results.append(SourceResult(row["source_id"], "ERROR", str(error)))
        return PipelineResult(tuple(results))

    if diarizer is None:
        diarizer = load_pyannote_diarizer(diarization_model)

    staged_sources: list[
        tuple[dict[str, str], Path, list[ClipMetadata], Path, Path, Path, Path, Path, Path]
    ] = []
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audio-processing-", dir=dataset_dir) as temporary_dir:
        stage_root = Path(temporary_dir)
        for row, media_path in candidates:
            try:
                metadata, staged_mp3, staged_turns, staged_clips, mp3_path, clips_dir = _process_one(
                    row, media_path, root, stage_root, resolved_ffmpeg, diarizer
                )
                source_turns_path = _output_paths(row, root, row["source_id"])[1]
                staged_sources.append(
                    (
                        row,
                        media_path,
                        metadata,
                        staged_mp3,
                        staged_turns,
                        staged_clips,
                        mp3_path,
                        source_turns_path,
                        clips_dir,
                    )
                )
            except Exception as error:  # noqa: BLE001 - isolate FFmpeg/Pyannote failures per source.
                detail = str(error) if isinstance(error, PipelineError) else type(error).__name__
                results.append(
                    SourceResult(row["source_id"], "ERROR", f"processing failed ({detail})")
                )

        if not staged_sources:
            return PipelineResult(tuple(results))

        updated_manifest = [dict(row) for row in rows]
        updated_metadata = list(metadata_rows)
        moves: list[tuple[Path, Path]] = []
        for (
            row,
            media_path,
            metadata,
            staged_mp3,
            staged_turns,
            staged_clips,
            mp3_path,
            turns_output,
            clips_dir,
        ) in staged_sources:
            mp3_path.parent.mkdir(parents=True, exist_ok=True)
            turns_output.parent.mkdir(parents=True, exist_ok=True)
            clips_dir.parent.mkdir(parents=True, exist_ok=True)
            moves.extend(
                (
                    (staged_mp3, mp3_path),
                    (staged_turns, turns_output),
                    (staged_clips, clips_dir),
                )
            )
            updated_metadata.extend(item.as_row() for item in metadata)
            for manifest_row in updated_manifest:
                if manifest_row["source_id"] == row["source_id"]:
                    manifest_row["original_video"] = media_path.relative_to(root).as_posix()
                    manifest_row["processing_status"] = "processed"
                    break

        created_outputs: list[Path] = []
        metadata_temp: Path | None = None
        manifest_temp: Path | None = None
        try:
            for staged_path, final_path in moves:
                if final_path.exists():
                    raise PipelineError(f"Output appeared during processing; refusing to overwrite: {final_path}")
                os.replace(staged_path, final_path)
                created_outputs.append(final_path)
            manifest_fields = list(rows[0].keys()) if rows else list(MANIFEST_FIELDS)
            metadata_temp = _write_rows_to_temp(metadata_path, CLIP_METADATA_FIELDS, updated_metadata)
            manifest_temp = _write_rows_to_temp(manifest_path, manifest_fields, updated_manifest)
            os.replace(metadata_temp, metadata_path)
            metadata_temp = None
            os.replace(manifest_temp, manifest_path)
            manifest_temp = None
        except Exception:
            if metadata_temp is not None:
                metadata_temp.unlink(missing_ok=True)
            if manifest_temp is not None:
                manifest_temp.unlink(missing_ok=True)
            for output in reversed(created_outputs):
                if output.is_dir():
                    shutil.rmtree(output, ignore_errors=True)
                else:
                    output.unlink(missing_ok=True)
            raise

        for row, _media_path, metadata, *_rest in staged_sources:
            results.append(
                SourceResult(row["source_id"], "PROCESSED", "completed conversion, full-recording diarization, clips, and QA", len(metadata))
            )

    return PipelineResult(tuple(results))


def load_pyannote_diarizer(model_id: str = "pyannote/speaker-diarization-3.1") -> Callable[[Path], list[SpeakerTurn]]:
    """Load the optional Pyannote diarizer lazily; HF_TOKEN is never displayed."""
    token = os.environ.get("HF_TOKEN")
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as error:
        raise PipelineError("Pyannote diarization dependencies are unavailable in this environment") from error

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
    except Exception as error:
        # Do not echo model-loader exception text, which can include sensitive request details.
        auth_hint = " Check cached model access or configure HF_TOKEN in the environment." if not token else ""
        raise PipelineError(
            f"Could not load Pyannote diarization model ({type(error).__name__}).{auth_hint}"
        ) from error

    def diarize(audio_path: Path) -> list[SpeakerTurn]:
        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as error:
            raise PipelineError(
                f"Could not decode complete recording for diarization ({type(error).__name__})"
            ) from error
        if audio.size == 0:
            raise PipelineError("Could not diarize an empty recording")
        waveform = torch.from_numpy(audio.T.copy())
        diarization_input = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "uri": audio_path.stem,
        }
        return _parse_diarization_result(pipeline(diarization_input))

    return diarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process authorized local podcast media; never downloads URL sources."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--media",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help="Explicit local input path; repeat once for each source to process.",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="FFmpeg executable; precedence is this option, FFMPEG_PATH, then system PATH.",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote diarization model ID; used only when there are local files to process.",
    )
    args = parser.parse_args(argv)
    try:
        overrides = _parse_media_arguments(args.media)
        result = run_pipeline(
            args.root,
            overrides,
            ffmpeg=args.ffmpeg,
            diarization_model=args.diarization_model,
        )
        print(result.format_text())
        return 1 if result.error_count else 0
    except (PipelineError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
