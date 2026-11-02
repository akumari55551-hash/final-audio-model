"""Evaluate the selected model once on a held-out, labeled split.

A threshold is either supplied explicitly or selected by F1 on development
scores. Evaluation labels never choose a threshold or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

from models.cnn_detector import LFCCCNNDetector
from training.audio_preprocessing import load_preprocessing_config, process_audio_file
from training.dataset import CLASS_NAMES, Recording, read_manifest_records
from training.evaluation_metrics import (
    compute_binary_metrics,
    save_confusion_svg,
    save_roc_svg,
    select_f1_threshold,
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a project training checkpoint with model_state_dict")
    if checkpoint.get("class_mapping") != {"bona_fide": 0, "spoof": 1}:
        raise ValueError("Checkpoint class mapping must be bona_fide=0, spoof=1")
    if checkpoint.get("training_metadata", {}).get("evaluation_split_used") is not False:
        raise ValueError("Checkpoint does not certify that evaluation was excluded from training/model selection")
    return checkpoint


def _load_model(checkpoint: dict[str, Any], device: torch.device) -> LFCCCNNDetector:
    settings = checkpoint.get("architecture_settings", {})
    if settings.get("name") != "LFCCCNNDetector":
        raise ValueError(f"Unsupported checkpoint architecture: {settings.get('name')!r}")
    model = LFCCCNNDetector(
        dropout=float(settings.get("dropout", 0.3)), seed=int(checkpoint.get("random_seed", 42))
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _read_evaluation_records(manifest: Path, split: str) -> list[Recording]:
    """Read a held-out partition. Deliberately separate from train/dev reader."""
    if split.casefold() in {"train", "dev", "development"}:
        raise ValueError("Refusing to evaluate a train or development split as final evaluation")
    if not manifest.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest}")
    records: list[Recording] = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"file_path", "label", "split", "source"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{manifest}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if (row.get("split") or "").strip().casefold() != split.casefold():
                continue
            raw_path = (row.get("file_path") or "").strip()
            if not raw_path:
                raise ValueError(f"{manifest}:{line_number}: empty file_path")
            try:
                label = int(row.get("label", ""))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{manifest}:{line_number}: invalid label") from exc
            if label not in CLASS_NAMES:
                raise ValueError(f"{manifest}:{line_number}: label must be 0 or 1")
            path = Path(raw_path).expanduser()
            path = path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{manifest}:{line_number}: audio file missing: {path}")
            source = (row.get("source") or "").strip()
            if not source:
                raise ValueError(f"{manifest}:{line_number}: source is empty")
            records.append(
                Recording(
                    path,
                    label,
                    split,
                    source,
                    (row.get("attack_type") or "").strip(),
                    (row.get("speaker_id") or "").strip(),
                    (row.get("utterance_id") or "").strip(),
                )
            )
    if not records:
        raise ValueError(f"No {split!r} records found in {manifest}")
    return records


def _scores_for_recording(
    model: LFCCCNNDetector,
    record: Recording,
    preprocessing,
    device: torch.device,
    window_batch_size: int,
) -> tuple[float, float, int, int]:
    """Return spoof score, forward ms/window, number of windows, samples/window."""
    windows, feature_array = process_audio_file(record.file_path, preprocessing)
    features = torch.from_numpy(np.array(feature_array[:, None, ...], dtype=np.float32, copy=True))
    outputs: list[Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, features.size(0), window_batch_size):
            outputs.append(model(features[offset : offset + window_batch_size].to(device)).float().cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    mean_logits = torch.cat(outputs).mean(dim=0)
    score = float(torch.softmax(mean_logits, dim=0)[1].item())
    return score, forward_seconds * 1000.0 / features.size(0), int(features.size(0)), int(windows.shape[1])


def _infer_scores(
    model: LFCCCNNDetector,
    records: list[Recording],
    preprocessing,
    device: torch.device,
    window_batch_size: int,
    title: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        score, latency_ms, window_count, samples_per_window = _scores_for_recording(
            model, record, preprocessing, device, window_batch_size
        )
        rows.append(
            {
                "file_path": str(record.file_path),
                "label": record.label,
                "source": record.source,
                "split": record.split,
                "attack_type": record.attack_type,
                "speaker_id": record.speaker_id,
                "utterance_id": record.utterance_id,
                "spoof_score": score,
                "inference_latency_ms_per_window": latency_ms,
                "window_count": window_count,
                "samples_per_window": samples_per_window,
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"{title}: {index}/{len(records)} records", flush=True)
    return rows


def _mean_absolute_amplitude(records: list[Recording], preprocessing) -> list[float]:
    """Minimal non-neural comparator; higher mean |waveform| means spoof score."""
    values: list[float] = []
    for record in records:
        windows, _ = process_audio_file(record.file_path, preprocessing)
        values.append(float(np.mean(np.abs(windows))))
    return values


def _metrics_by_attack(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["attack_type"]:
            grouped[str(row["attack_type"])].append(row)
    result: dict[str, Any] = {}
    for attack, members in sorted(grouped.items()):
        result[attack] = compute_binary_metrics(
            [int(row["label"]) for row in members],
            [float(row["spoof_score"]) for row in members],
            threshold,
        ).to_dict()
    return result


def _write_predictions(path: Path, rows: list[dict[str, Any]], threshold: float) -> None:
    fields = (
        "file_path", "label", "source", "split", "attack_type", "speaker_id",
        "utterance_id", "spoof_score", "predicted_label", "threshold",
        "inference_latency_ms_per_window", "window_count", "samples_per_window",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "predicted_label": int(row["spoof_score"] >= threshold), "threshold": threshold})


def evaluate(
    checkpoint_path: Path,
    evaluation_manifest: Path,
    *,
    evaluation_split: str,
    output_dir: Path,
    dev_manifest: Path | None = None,
    dev_split: str = "dev",
    threshold: float | None = None,
    device_name: str = "auto",
    window_batch_size: int = 16,
    external_dataset: bool = False,
) -> dict[str, Any]:
    """Evaluate a project best checkpoint; dev is used only for threshold selection."""
    if evaluation_split.casefold() in {"train", "dev", "development"}:
        raise ValueError("Evaluation split must be held out and cannot be train/dev")
    if window_batch_size < 1:
        raise ValueError("window_batch_size must be positive")
    if dev_manifest is None:
        raise ValueError(
            "--dev-manifest is required to select the CNN threshold and independently calibrate the baseline"
        )
    device = _resolve_device(device_name)
    checkpoint_path = checkpoint_path.resolve()
    evaluation_manifest = evaluation_manifest.resolve()
    dev_manifest = dev_manifest.resolve() if dev_manifest else None
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_config = checkpoint.get("preprocessing_config")
    if not isinstance(checkpoint_config, dict):
        raise TypeError("Checkpoint does not contain preprocessing configuration mapping")
    config_path = output_dir / ".checkpoint_preprocessing.yaml"
    config_path.write_text(yaml.safe_dump(checkpoint_config, sort_keys=False), encoding="utf-8")
    try:
        preprocessing = load_preprocessing_config(config_path)
    finally:
        config_path.unlink(missing_ok=True)
    model = _load_model(checkpoint, device)

    dev_selection: dict[str, Any] = {
        "method": "caller-supplied threshold; verify it was selected using development data"
    }
    baseline_dev_cache: tuple[list[Recording], list[float]] | None = None
    dev_records = read_manifest_records(dev_manifest, dev_split)
    if threshold is None:
        dev_rows = _infer_scores(model, dev_records, preprocessing, device, window_batch_size, "Development")
        threshold, selection_values = select_f1_threshold(
            [int(row["label"]) for row in dev_rows], [float(row["spoof_score"]) for row in dev_rows]
        )
        dev_selection = {
            "method": "maximum spoof F1; ties choose the higher threshold",
            "source_manifest": str(dev_manifest),
            "split": dev_split,
            **selection_values,
        }
        (output_dir / "dev_threshold_selection.json").write_text(
            json.dumps({"threshold": threshold, "selection": dev_selection}, indent=2) + "\n",
            encoding="utf-8",
        )
        baseline_dev_cache = (
            dev_records,
            _mean_absolute_amplitude(dev_records, preprocessing),
        )
    else:
        baseline_dev_cache = (dev_records, _mean_absolute_amplitude(dev_records, preprocessing))

    records = _read_evaluation_records(evaluation_manifest, evaluation_split)
    expected_source = "WaveFake" if external_dataset else "ASVspoof2019_LA"
    record_sources = {record.source for record in records}
    if external_dataset:
        if "WaveFake" not in record_sources or any(
            record.source != "WaveFake" and record.label != 0 for record in records
        ):
            raise ValueError(
                "External evaluation must include WaveFake spoof rows; any non-WaveFake paired reference rows must be bona fide"
            )
    elif record_sources != {"ASVspoof2019_LA"}:
        raise ValueError("Primary evaluation expects only ASVspoof2019_LA records")
    if {record.label for record in records} != {0, 1}:
        raise ValueError("This evaluation requires both classes; generated-only WaveFake cannot yield binary metrics without bona-fide references")

    wall_start = time.perf_counter()
    predictions = _infer_scores(model, records, preprocessing, device, window_batch_size, "Evaluation")
    total_wall_seconds = time.perf_counter() - wall_start
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row["spoof_score"]) for row in predictions]
    cnn_metrics = compute_binary_metrics(labels, scores, float(threshold))

    baseline_scores = _mean_absolute_amplitude(records, preprocessing)
    baseline_dev_records, baseline_dev_scores = baseline_dev_cache
    baseline_threshold, baseline_threshold_info = select_f1_threshold(
        [record.label for record in baseline_dev_records], baseline_dev_scores
    )
    baseline_threshold_info = {
        "method": "maximum spoof F1 on development; ties choose higher threshold",
        **baseline_threshold_info,
    }
    baseline_metrics = compute_binary_metrics(labels, baseline_scores, baseline_threshold)

    _write_predictions(output_dir / "predictions.csv", predictions, float(threshold))
    if external_dataset:
        audio_caveat = (
            "WaveFake is an external generated-audio collection (16-bit PCM WAV, source corpora include English LJSpeech and Japanese JSUT; original record does not redistribute reference corpora). Conditions differ from ASVspoof LA's VCTK-derived challenge data. This run requires labeled bona-fide reference records in the supplied external evaluation manifest."
        )
    else:
        audio_caveat = "Official ASVspoof 2019 Logical Access evaluation partition; conclusions are limited to this protocol and its conditions."

    save_roc_svg(
        str(output_dir / "roc.svg"),
        {
            "LFCC CNN": (labels, scores),
            "Mean absolute amplitude": (labels, baseline_scores),
        },
        title="WaveFake external evaluation ROC" if external_dataset else "ASVspoof 2019 LA evaluation ROC",
    )
    save_confusion_svg(str(output_dir / "confusion_matrix.svg"), cnn_metrics.confusion_matrix, "LFCC CNN confusion matrix")
    save_confusion_svg(str(output_dir / "baseline_confusion_matrix.svg"), baseline_metrics.confusion_matrix, "Amplitude baseline confusion matrix")

    latency_values = np.asarray([row["inference_latency_ms_per_window"] for row in predictions], dtype=np.float64)
    report: dict[str, Any] = {
        "report_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "dataset": {
            "source": expected_source,
            "source_values": sorted(record_sources),
            "split": evaluation_split,
            "external_dataset": external_dataset,
            "record_count": len(records),
            "audio_conditions": audio_caveat,
            "attack_types": sorted({record.attack_type for record in records if record.attack_type}),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "selection_metric": checkpoint.get("training_metadata", {}).get("selection_metric"),
            "threshold": float(threshold),
            "threshold_selection": dev_selection,
            "score_convention": "CNN spoof score is softmax(mean of per-window raw logits)[class index 1]; predict spoof when score >= threshold.",
            "evaluation_used_for_selection": False,
        },
        "metrics": cnn_metrics.to_dict(),
        "metrics_by_attack_type": _metrics_by_attack(predictions, float(threshold)),
        "simple_non_neural_baseline": {
            "name": "mean absolute waveform amplitude",
            "score_convention": "higher mean absolute amplitude indicates spoof",
            "threshold": float(baseline_threshold),
            "threshold_selection": baseline_threshold_info,
            "metrics": baseline_metrics.to_dict(),
            "metrics_by_attack_type": _metrics_by_attack(
                [dict(row, spoof_score=score) for row, score in zip(predictions, baseline_scores)],
                float(baseline_threshold),
            ),
            "caveat": "Sanity-check comparator only; not a validated anti-spoofing detector.",
        },
        "latency": {
            "model_forward_ms_per_window_mean": float(latency_values.mean()),
            "model_forward_ms_per_window_median": float(np.median(latency_values)),
            "model_forward_ms_per_window_p95": float(np.percentile(latency_values, 95)),
            "scope": "CUDA-synchronized model scoring time including tensor transfer and forward passes; excludes decode/resample/LFCC extraction, audio buffering, and network transport.",
            "offline_evaluation_wall_seconds_including_preprocessing": total_wall_seconds,
            "audio_buffering_latency": "not measured in offline file evaluation",
        },
        "definitions": {
            "positive_class": "spoof (label 1)",
            "confusion_matrix": "rows=true [bona_fide, spoof], columns=predicted [bona_fide, spoof]",
            "fpr": "FP / (FP + TN)",
            "fnr": "FN / (FN + TP)",
            "roc_auc": "trapezoidal ROC area from continuous spoof scores; null if either class is absent",
            "eer": "linearly interpolated ROC operating point where FPR equals FNR (1-TPR); mean of interpolated rates; null if either class is absent",
        },
        "artifacts": {
            "predictions_csv": str(output_dir / "predictions.csv"),
            "roc_svg": str(output_dir / "roc.svg"),
            "confusion_matrix_svg": str(output_dir / "confusion_matrix.svg"),
            "baseline_confusion_matrix_svg": str(output_dir / "baseline_confusion_matrix.svg"),
        },
        "generalization_statement": (
            "External WaveFake result only; no unseen-generator generalization claim is made."
            if external_dataset
            else "ASVspoof 2019 LA evaluation result only; no broader generalization claim is made."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Threshold (dev-selected): {threshold:.6f}", flush=True)
    print(f"Evaluation F1={cnn_metrics.f1:.4f}, EER={cnn_metrics.equal_error_rate}, AUC={cnn_metrics.roc_auc}", flush=True)
    print(f"Wrote {output_dir / 'metrics.json'} and predictions.csv", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-split", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--threshold", type=float, help="Preselected using development data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--external-dataset", action="store_true", help="Mark WaveFake external evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluate(
            args.checkpoint,
            args.evaluation_manifest,
            evaluation_split=args.evaluation_split,
            output_dir=args.output_dir,
            dev_manifest=args.dev_manifest,
            dev_split=args.dev_split,
            threshold=args.threshold,
            device_name=args.device,
            window_batch_size=args.window_batch_size,
            external_dataset=args.external_dataset,
        )
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        print(f"ERROR: evaluation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
