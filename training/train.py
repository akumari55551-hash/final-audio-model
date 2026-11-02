"""Supervised training for the custom LFCC CNN.

Only manifest rows marked ``train`` and ``dev`` are loaded. Evaluation rows
are never read or used for checkpoint selection, scheduling, or early stopping.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.cnn_detector import LFCCCNNDetector, count_trainable_parameters
from models.lfcc import set_reproducible_seed
from training.audio_preprocessing import load_preprocessing_config
from training.dataset import ManifestAudioDataset, class_counts
from training.metrics import ClassificationMetrics, classification_metrics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not load config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"Training config {path} must contain a YAML mapping")
    return loaded


def _resolve_path(value: str | Path, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(f"Unsupported device setting {requested!r}") from exc


def _validate_class_coverage(name: str, records: ManifestAudioDataset) -> dict[int, int]:
    counts = class_counts(records.records)
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        names = ["bona_fide" if label == 0 else "spoof" for label in missing]
        raise ValueError(f"{name} split must contain both classes; missing: {names}; counts={counts}")
    return counts


def _stratified_sampler(dataset: ManifestAudioDataset, seed: int) -> WeightedRandomSampler:
    counts = class_counts(dataset.records)
    sample_weights = [1.0 / counts[record.label] for record in dataset.records]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _class_weights(dataset: ManifestAudioDataset, device: torch.device) -> Tensor:
    counts = class_counts(dataset.records)
    total = sum(counts.values())
    return torch.tensor(
        [total / (2.0 * counts[label]) for label in (0, 1)],
        dtype=torch.float32,
        device=device,
    )


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    epoch: int,
    total_epochs: int,
) -> ClassificationMetrics:
    model.train()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    loss_total = 0.0
    sample_count = 0
    progress_interval = max(1, math.ceil(len(loader) / 20))
    for batch_index, (features, labels) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, mixed_precision):
            logits = model(features)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count = int(labels.size(0))
        loss_total += float(loss.detach().float().item()) * count
        sample_count += count
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.detach().argmax(dim=1).cpu().tolist())
        if batch_index % progress_interval == 0 or batch_index == len(loader):
            print(
                f"  train epoch {epoch}/{total_epochs} batch {batch_index}/{len(loader)} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    if sample_count == 0:
        raise RuntimeError("Training loader yielded no batches")
    return classification_metrics(labels_all, predictions_all, loss_total / sample_count)


@torch.inference_mode()
def _evaluate_dev(
    model: nn.Module,
    dataset: ManifestAudioDataset,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool,
    window_batch_size: int,
) -> ClassificationMetrics:
    """Evaluate all windows but score each dev recording once using mean logits."""
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    loss_total = 0.0
    for index, record in enumerate(dataset.records):
        all_windows, label_tensor = dataset.get_all_windows(index)
        window_logits: list[Tensor] = []
        for start in range(0, all_windows.size(0), window_batch_size):
            window_batch = all_windows[start : start + window_batch_size].to(device)
            with _autocast_context(device, mixed_precision):
                logits = model(window_batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite dev logits for {record.file_path}")
            window_logits.append(logits.float().cpu())
        recording_logits = torch.cat(window_logits, dim=0).mean(dim=0, keepdim=True)
        label = label_tensor.reshape(1).to(device)
        loss = criterion(recording_logits.to(device), label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite development loss for {record.file_path}")
        loss_total += float(loss.item())
        labels_all.append(int(label_tensor.item()))
        predictions_all.append(int(recording_logits.argmax(dim=1).item()))
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(
                f"  dev progress {index + 1}/{len(dataset)} recordings",
                flush=True,
            )
    return classification_metrics(labels_all, predictions_all, loss_total / len(dataset))


def _metric_line(name: str, metrics: ClassificationMetrics) -> str:
    matrix = metrics.confusion_matrix
    return (
        f"{name}: loss={metrics.loss:.5f} precision_macro={metrics.precision:.4f} "
        f"recall_macro={metrics.recall:.4f} f1_macro={metrics.f1:.4f} "
        f"spoof(P/R/F1)={metrics.spoof_precision:.4f}/{metrics.spoof_recall:.4f}/"
        f"{metrics.spoof_f1:.4f} confusion=[[{matrix[0][0]}, {matrix[0][1]}], "
        f"[{matrix[1][0]}, {matrix[1][1]}]]"
    )


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_file)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: LFCCCNNDetector,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    expected_config: dict[str, Any],
    expected_dropout: float,
    expected_seed: int,
    expected_manifests: dict[str, str],
) -> tuple[int, float, float, int, list[dict[str, Any]]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is not a compatible project training checkpoint; missing {sorted(missing)}")
    if checkpoint.get("class_mapping") != {"bona_fide": 0, "spoof": 1}:
        raise ValueError("Checkpoint class mapping does not match this trainer")
    saved_config = checkpoint.get("preprocessing_config", {})
    if any(saved_config.get(section) != expected_config.get(section) for section in ("audio", "features")):
        raise ValueError("Checkpoint audio/LFCC preprocessing settings differ from the requested config")
    saved_model = saved_config.get("model", {})
    expected_model = expected_config.get("model", {})
    model_keys = ("architecture", "initialization", "pretrained_weights", "dropout", "classes")
    if any(saved_model.get(key) != expected_model.get(key) for key in model_keys):
        raise ValueError("Checkpoint model architecture/class settings differ from the requested config")
    if int(checkpoint.get("random_seed", -1)) != expected_seed:
        raise ValueError("Checkpoint random seed differs from the requested training seed")
    architecture_settings = checkpoint.get("architecture_settings", {})
    if architecture_settings.get("name") != "LFCCCNNDetector" or architecture_settings.get(
        "dropout"
    ) != expected_dropout:
        raise ValueError("Checkpoint architecture settings do not match the requested model")
    saved_manifests = checkpoint.get("training_metadata", {}).get("manifests", {})
    if saved_manifests != expected_manifests:
        raise ValueError("Checkpoint train/dev manifest paths differ from the requested manifests")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_dev_f1_macro", -math.inf)),
        float(checkpoint.get("best_dev_loss", math.inf)),
        int(checkpoint.get("epochs_without_improvement", 0)),
        list(checkpoint.get("history", [])),
    )


def train(
    config_path: Path,
    *,
    train_manifest: Path | None = None,
    dev_manifest: Path | None = None,
    resume: Path | None = None,
    smoke: bool = False,
    smoke_limit_per_class: int = 4,
) -> dict[str, Any]:
    """Train from train/dev manifests and return run metadata/history."""
    raw_config = _load_yaml(config_path)
    project = raw_config.get("project", {})
    model_config = raw_config.get("model", {})
    training_config = raw_config.get("training", {})
    if not isinstance(project, dict) or not isinstance(model_config, dict) or not isinstance(
        training_config, dict
    ):
        raise TypeError("project, model, and training configuration sections must be mappings")
    if model_config.get("pretrained_weights", False) or model_config.get("initialization", "random") != "random":
        raise ValueError("This trainer supports randomly initialized models only; pretrained weights are prohibited")

    seed = int(training_config.get("seed", project.get("seed", model_config.get("seed", 42))))
    if seed < 0:
        raise ValueError("seed must be non-negative")
    set_reproducible_seed(seed, deterministic_algorithms=bool(training_config.get("deterministic", False)))

    shared_manifest = _resolve_path(training_config.get("manifest_path", "data/manifests/asvspoof2019_la.csv"))
    train_path = _resolve_path(train_manifest) if train_manifest else _resolve_path(
        training_config.get("train_manifest", shared_manifest)
    )
    dev_path = _resolve_path(dev_manifest) if dev_manifest else _resolve_path(
        training_config.get("dev_manifest", shared_manifest)
    )
    preprocessing_path = _resolve_path(raw_config.get("preprocessing_config", "configs/baseline.yaml"))
    preprocessing = load_preprocessing_config(preprocessing_path)

    smoke_limit = smoke_limit_per_class if smoke else None
    training_data = ManifestAudioDataset(
        train_path, "train", preprocessing, seed=seed, per_class_limit=smoke_limit
    )
    dev_data = ManifestAudioDataset(
        dev_path, "dev", preprocessing, seed=seed, per_class_limit=smoke_limit
    )
    train_counts = _validate_class_coverage("train", training_data)
    dev_counts = _validate_class_coverage("dev", dev_data)

    batch_size = int(training_config.get("batch_size", 16))
    epochs = 1 if smoke else int(training_config.get("epochs", 30))
    if smoke:
        batch_size = min(batch_size, max(2, len(training_data)))
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    num_workers = int(training_config.get("num_workers", 0))
    if batch_size < 1 or epochs < 1 or num_workers < 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid batch size, epoch count, worker count, learning rate, or weight decay")
    imbalance_strategy = str(training_config.get("imbalance_strategy", "weighted_loss"))
    if imbalance_strategy not in {"none", "weighted_loss", "weighted_sampler"}:
        raise ValueError("imbalance_strategy must be none, weighted_loss, or weighted_sampler")

    requested_device = str(training_config.get("device", project.get("device", "auto")))
    device = _resolve_device(requested_device)
    mixed_precision = bool(training_config.get("mixed_precision", False)) and device.type == "cuda"
    if bool(training_config.get("mixed_precision", False)) and device.type != "cuda":
        print("INFO: Mixed precision disabled because the active device is CPU.", flush=True)

    dropout = float(model_config.get("dropout", 0.3))
    model = LFCCCNNDetector(dropout=dropout, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler_config = training_config.get("scheduler", {})
    if not isinstance(scheduler_config, dict):
        raise TypeError("training.scheduler must be a mapping")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 2)),
        min_lr=float(scheduler_config.get("min_lr", 1.0e-6)),
    )
    validation_window_batch_size = int(
        training_config.get("validation_window_batch_size", batch_size)
    )
    if validation_window_batch_size < 1:
        raise ValueError("validation_window_batch_size must be positive")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    loss_weights = _class_weights(training_data, device) if imbalance_strategy == "weighted_loss" else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    sampler = _stratified_sampler(training_data, seed) if imbalance_strategy == "weighted_sampler" else None
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )

    checkpoint_dir = _resolve_path(training_config.get("checkpoint_dir", "checkpoints"))
    if smoke:
        if resume is not None:
            raise ValueError("Do not resume a smoke test; use resume only for full training runs")
        checkpoint_dir = checkpoint_dir / "smoke"
    elif resume is not None and _resolve_path(resume).parent != checkpoint_dir:
        raise ValueError(
            "Resume checkpoints must be in the configured checkpoint directory so the run can update last.pt/best.pt safely"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_dev_f1_macro = -math.inf
    best_dev_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        start_epoch, best_dev_f1_macro, best_dev_loss, epochs_without_improvement, history = _load_resume_checkpoint(
            _resolve_path(resume),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            raw_config,
            dropout,
            seed,
            {"train": str(train_path), "dev": str(dev_path)},
        )
        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch - 1}, not before configured epochs={epochs}"
            )

    print(f"Device: {device}; mixed_precision={mixed_precision}; seed={seed}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Training recordings: {len(training_data)}; class counts={train_counts}", flush=True)
    print(f"Development recordings: {len(dev_data)}; class counts={dev_counts}", flush=True)
    print(f"Imbalance strategy: {imbalance_strategy}; trainable parameters={count_trainable_parameters(model)}", flush=True)
    if smoke:
        print(
            "SMOKE TEST: capped real manifest records and one training epoch; "
            "metrics are pipeline checks, not benchmark/model-performance claims.",
            flush=True,
        )

    patience = int(training_config.get("early_stopping_patience", 6))
    if patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    run_metadata: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "architecture": {
            "name": "LFCCCNNDetector",
            "initialization": "random",
            "pretrained_weights": False,
            "dropout": dropout,
            "input_shape": "[batch, 1, n_lfcc, frames]",
            "class_mapping": {"bona_fide": 0, "spoof": 1},
            "trainable_parameters": count_trainable_parameters(model),
        },
        "manifests": {"train": str(train_path), "dev": str(dev_path)},
        "class_counts": {
            "train": {str(key): value for key, value in train_counts.items()},
            "dev": {str(key): value for key, value in dev_counts.items()},
        },
        "imbalance_strategy": imbalance_strategy,
        "smoke_test": smoke,
        "selection_metric": "development macro F1; tie-break by lower development loss",
        "evaluation_split_used": False,
    }

    for epoch in range(start_epoch, epochs + 1):
        training_data.set_epoch(epoch)
        train_metrics = _run_training_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            mixed_precision,
            epoch,
            epochs,
        )
        dev_metrics = _evaluate_dev(
            model,
            dev_data,
            criterion,
            device,
            mixed_precision,
            validation_window_batch_size,
        )
        scheduler.step(dev_metrics.f1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(_metric_line("train", train_metrics), flush=True)
        print(_metric_line("dev", dev_metrics), flush=True)
        print(f"learning_rate={current_lr:.8g}", flush=True)
        epoch_entry = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics.as_dict(),
            "dev": dev_metrics.as_dict(),
        }
        history.append(epoch_entry)

        improved = dev_metrics.f1 > best_dev_f1_macro or (
            math.isclose(dev_metrics.f1, best_dev_f1_macro) and dev_metrics.loss < best_dev_loss
        )
        if improved:
            best_dev_f1_macro = dev_metrics.f1
            best_dev_loss = dev_metrics.loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_payload: dict[str, Any] = {
            "format_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_dev_f1_macro": best_dev_f1_macro,
            "best_dev_loss": best_dev_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "architecture_settings": run_metadata["architecture"],
            "preprocessing_config": raw_config,
            "preprocessing_config_path": str(preprocessing_path),
            "class_mapping": {"bona_fide": 0, "spoof": 1},
            "random_seed": seed,
            "training_metadata": run_metadata,
        }
        _atomic_torch_save(checkpoint_payload, last_checkpoint)
        if improved:
            _atomic_torch_save(checkpoint_payload, best_checkpoint)
            print(f"Selected new best checkpoint using dev macro F1: {best_checkpoint}", flush=True)
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs without dev improvement.", flush=True)
            break

    metrics_path = checkpoint_dir / "history.json"
    metrics_path.write_text(
        json.dumps({"metadata": run_metadata, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Last checkpoint: {last_checkpoint}", flush=True)
    print(f"Best checkpoint: {best_checkpoint}", flush=True)
    print(f"Training history: {metrics_path}", flush=True)
    if not best_checkpoint.is_file():
        raise RuntimeError("Training ended without a selected best checkpoint")
    return {"metadata": run_metadata, "history": history, "best_checkpoint": str(best_checkpoint)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-manifest", type=Path, help="Overrides configured training manifest")
    parser.add_argument("--dev-manifest", type=Path, help="Overrides configured development manifest")
    parser.add_argument("--resume", type=Path, help="Resume from this project's own last.pt checkpoint")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one capped epoch on real manifest records; saves only under checkpoints/smoke",
    )
    parser.add_argument("--smoke-limit-per-class", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_limit_per_class < 1:
        print("ERROR: --smoke-limit-per-class must be positive", file=sys.stderr)
        return 2
    try:
        train(
            args.config.resolve(),
            train_manifest=args.train_manifest,
            dev_manifest=args.dev_manifest,
            resume=args.resume,
            smoke=args.smoke,
            smoke_limit_per_class=args.smoke_limit_per_class,
        )
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        print(f"ERROR: training stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
