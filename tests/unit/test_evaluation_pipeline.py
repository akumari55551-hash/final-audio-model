"""Evaluation integration test with synthetic local fixtures only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from models.cnn_detector import LFCCCNNDetector
from training.evaluate import evaluate

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"


def _write_wave(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(path, 0.2 * np.sin(2 * np.pi * frequency * time), sample_rate, subtype="FLOAT")


def _manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("file_path", "label", "split", "source", "attack_type", "speaker_id", "utterance_id")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluation_emits_artifacts_and_uses_dev_threshold_only(tmp_path: Path) -> None:
    """Generated fixture scores/artifacts exercise code only, not real results."""
    audio_root = tmp_path / "audio"
    dev_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for split, target_rows in (("dev", dev_rows), ("eval", eval_rows)):
        for label, frequency, attack in ((0, 220.0, ""), (1, 880.0, "A-fixture")):
            audio = audio_root / f"{split}-{label}.wav"
            _write_wave(audio, frequency)
            target_rows.append(
                {
                    "file_path": str(audio),
                    "label": str(label),
                    "split": split,
                    "source": "ASVspoof2019_LA",
                    "attack_type": attack,
                    "speaker_id": f"{split}-{label}",
                    "utterance_id": f"synthetic-{split}-{label}",
                }
            )
    dev_manifest = tmp_path / "dev.csv"
    eval_manifest = tmp_path / "eval.csv"
    _manifest(dev_manifest, dev_rows)
    _manifest(eval_manifest, eval_rows)

    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    model = LFCCCNNDetector(dropout=float(config["model"]["dropout"]), seed=42)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture_settings": {"name": "LFCCCNNDetector", "dropout": 0.3},
            "random_seed": 42,
            "class_mapping": {"bona_fide": 0, "spoof": 1},
            "preprocessing_config": config,
            "training_metadata": {"evaluation_split_used": False, "selection_metric": "dev macro F1"},
            "epoch": 1,
        },
        checkpoint_path,
    )

    output = tmp_path / "evaluation"
    report = evaluate(
        checkpoint_path,
        eval_manifest,
        evaluation_split="eval",
        output_dir=output,
        dev_manifest=dev_manifest,
        device_name="cpu",
        window_batch_size=2,
    )
    assert report["checkpoint"]["evaluation_used_for_selection"] is False
    assert report["dataset"]["record_count"] == 2
    assert report["metrics"]["support"] == 2
    assert report["metrics_by_attack_type"]["A-fixture"]["support"] == 1
    assert (output / "predictions.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "dev_threshold_selection.json").is_file()
    assert (output / "roc.svg").is_file()
    assert (output / "confusion_matrix.svg").is_file()
    assert (output / "baseline_confusion_matrix.svg").is_file()


def test_refuses_dev_partition_as_final_evaluation(tmp_path: Path) -> None:
    try:
        evaluate(
            tmp_path / "missing.pt",
            tmp_path / "missing.csv",
            evaluation_split="dev",
            output_dir=tmp_path / "out",
            threshold=0.5,
        )
    except ValueError as exc:
        assert "cannot be train/dev" in str(exc)
    else:
        raise AssertionError("dev partition should not be accepted as final evaluation")


def test_wavefake_external_eval_accepts_only_documented_bona_fide_reference_rows(tmp_path: Path) -> None:
    """External fixture verifies WaveFake spoof plus separately sourced bona fide handling."""
    audio_root = tmp_path / "audio"
    dev_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for split, target_rows in (("dev", dev_rows), ("external_eval", eval_rows)):
        for label, frequency, source in (
            (0, 220.0, "ASVspoof2019_LA"),
            (1, 880.0, "WaveFake"),
        ):
            audio = audio_root / f"{split}-{label}.wav"
            _write_wave(audio, frequency)
            target_rows.append(
                {
                    "file_path": str(audio),
                    "label": str(label),
                    "split": split,
                    "source": source,
                    "attack_type": "external-generator" if label else "",
                    "speaker_id": f"{split}-{label}",
                    "utterance_id": f"synthetic-{split}-{label}",
                }
            )
    dev_manifest = tmp_path / "dev.csv"
    external_manifest = tmp_path / "external.csv"
    _manifest(dev_manifest, dev_rows)
    _manifest(external_manifest, eval_rows)
    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    model = LFCCCNNDetector(dropout=0.3, seed=42)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture_settings": {"name": "LFCCCNNDetector", "dropout": 0.3},
            "random_seed": 42,
            "class_mapping": {"bona_fide": 0, "spoof": 1},
            "preprocessing_config": config,
            "training_metadata": {"evaluation_split_used": False},
            "epoch": 1,
        },
        checkpoint_path,
    )
    report = evaluate(
        checkpoint_path,
        external_manifest,
        evaluation_split="external_eval",
        output_dir=tmp_path / "external-output",
        dev_manifest=dev_manifest,
        device_name="cpu",
        external_dataset=True,
    )
    assert report["dataset"]["external_dataset"] is True
    assert report["dataset"]["source_values"] == ["ASVspoof2019_LA", "WaveFake"]
