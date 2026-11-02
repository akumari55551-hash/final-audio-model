"""Synthetic integration checks; generated fixture metrics are not benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from models.cnn_detector import LFCCCNNDetector
from training.audio_preprocessing import load_preprocessing_config
from training.dataset import ManifestAudioDataset, read_manifest_records
from training.metrics import classification_metrics
from training.train import _load_resume_checkpoint, train

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"


def _write_audio(path: Path, frequency: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = 0.2 * np.sin(2 * np.pi * frequency * time)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("file_path", "label", "split", "source", "attack_type", "speaker_id", "utterance_id"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_reads_requested_split_only_and_ignores_eval_row(tmp_path: Path) -> None:
    train_a = tmp_path / "train-a.wav"
    dev_a = tmp_path / "dev-a.wav"
    _write_audio(train_a, 220)
    _write_audio(dev_a, 330)
    manifest = tmp_path / "combined.csv"
    _write_manifest(
        manifest,
        [
            {"file_path": str(train_a), "label": "0", "split": "train", "source": "fixture"},
            {"file_path": str(dev_a), "label": "1", "split": "dev", "source": "fixture"},
            # This deliberately nonexistent evaluation path proves that the
            # training dataset does not open/evaluate evaluation rows.
            {"file_path": str(tmp_path / "never-open-eval.wav"), "label": "1", "split": "eval", "source": "fixture"},
        ],
    )
    records = read_manifest_records(manifest, "train")
    assert len(records) == 1
    assert records[0].file_path == train_a.resolve()
    with pytest.raises(ValueError, match="Only train/dev"):
        read_manifest_records(manifest, "eval")


def test_dataset_uses_shared_preprocessing_and_returns_tensor_shape(tmp_path: Path) -> None:
    audio = tmp_path / "bonafide.wav"
    _write_audio(audio, 440)
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [
            {"file_path": str(audio), "label": "0", "split": "train", "source": "fixture"},
            {"file_path": str(audio), "label": "1", "split": "train", "source": "fixture", "utterance_id": "second"},
        ],
    )
    preprocessing = load_preprocessing_config(BASELINE_CONFIG)
    dataset = ManifestAudioDataset(manifest, "train", preprocessing, seed=2)
    feature, label = dataset[0]
    windows, all_label = dataset.get_all_windows(0)
    assert feature.shape == (1, 40, 198)
    assert windows.shape == (1, 1, 40, 198)
    assert label.dtype == torch.long
    assert all_label.item() == 0


def test_metrics_confusion_matrix_and_spoof_scores() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], mean_loss=0.5)
    assert metrics.confusion_matrix == [[1, 1], [0, 2]]
    assert metrics.spoof_precision == pytest.approx(2 / 3)
    assert metrics.spoof_recall == pytest.approx(1.0)
    assert metrics.spoof_f1 == pytest.approx(0.8)
    assert metrics.support == 4


def test_smoke_training_checkpoint_and_metadata_use_synthetic_fixtures(tmp_path: Path) -> None:
    """One tiny synthetic-fixture epoch tests the pipeline, not detection quality."""
    audio_dir = tmp_path / "audio"
    rows: list[dict[str, str]] = []
    for split in ("train", "dev"):
        for label, frequency in ((0, 220.0), (1, 880.0)):
            audio = audio_dir / f"{split}-{label}.wav"
            _write_audio(audio, frequency)
            rows.append(
                {
                    "file_path": str(audio),
                    "label": str(label),
                    "split": split,
                    "source": "SYNTHETIC_TEST_FIXTURE_NOT_BENCHMARK_DATA",
                    "attack_type": "fixture-generator" if label else "",
                    "speaker_id": f"{split}-{label}",
                    "utterance_id": f"fixture-{split}-{label}",
                }
            )
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)

    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    config["training"].update(
        {
            "train_manifest": str(manifest_path),
            "dev_manifest": str(manifest_path),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
            "mixed_precision": False,
            "imbalance_strategy": "weighted_loss",
            "early_stopping_patience": 1,
            "validation_window_batch_size": 2,
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train(config_path, smoke=True, smoke_limit_per_class=2)
    best_path = Path(result["best_checkpoint"])
    assert best_path.is_file()
    assert (best_path.parent / "last.pt").is_file()
    assert (best_path.parent / "history.json").is_file()
    assert result["metadata"]["smoke_test"] is True
    assert result["metadata"]["evaluation_split_used"] is False
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    assert checkpoint["class_mapping"] == {"bona_fide": 0, "spoof": 1}
    assert checkpoint["architecture_settings"]["initialization"] == "random"
    assert checkpoint["random_seed"] == 42

    model = LFCCCNNDetector(dropout=0.3, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch, best_f1, best_loss, stale_epochs, history = _load_resume_checkpoint(
        best_path,
        model,
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        config,
        0.3,
        42,
        {"train": str(manifest_path.resolve()), "dev": str(manifest_path.resolve())},
    )
    assert start_epoch == 2
    assert best_f1 == pytest.approx(checkpoint["best_dev_f1_macro"])
    assert best_loss == pytest.approx(checkpoint["best_dev_loss"])
    assert stale_epochs == 0
    assert len(history) == 1

    changed_epoch_config = yaml.safe_load(yaml.safe_dump(config))
    changed_epoch_config["training"]["epochs"] = 10
    resumed, *_ = _load_resume_checkpoint(
        best_path,
        model,
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        changed_epoch_config,
        0.3,
        42,
        {"train": str(manifest_path.resolve()), "dev": str(manifest_path.resolve())},
    )
    assert resumed == 2

    incompatible_config = yaml.safe_load(yaml.safe_dump(config))
    incompatible_config["features"]["n_coefficients"] = 20
    with pytest.raises(ValueError, match="preprocessing settings"):
        _load_resume_checkpoint(
            best_path,
            model,
            optimizer,
            scheduler,
            scaler,
            torch.device("cpu"),
            incompatible_config,
            0.3,
            42,
            {"train": str(manifest_path.resolve()), "dev": str(manifest_path.resolve())},
        )
