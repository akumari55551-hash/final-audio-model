"""Trusted local checkpoint inference for live mono audio windows."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from models.cnn_detector import LFCCCNNDetector
from training.audio_preprocessing import (
    PreprocessingConfig,
    load_preprocessing_config,
    process_waveform,
)


class ModelLoadError(RuntimeError):
    """Raised when the local checkpoint cannot be safely loaded or validated."""


@dataclass(frozen=True)
class AudioQualityStatus:
    status: str
    rms: float
    peak: float
    clipped_fraction: float
    dropped_audio_chunks: int
    dropped_audio_samples: int
    skipped_inference_windows: int
    details: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rms": self.rms,
            "peak": self.peak,
            "clipped_fraction": self.clipped_fraction,
            "dropped_audio_chunks": self.dropped_audio_chunks,
            "dropped_audio_samples": self.dropped_audio_samples,
            "skipped_inference_windows": self.skipped_inference_windows,
            "details": list(self.details),
        }


def assess_audio_quality(
    waveform: np.ndarray,
    *,
    dropped_audio_chunks: int = 0,
    dropped_audio_samples: int = 0,
    skipped_inference_windows: int = 0,
    silence_rms_threshold: float = 1.0e-4,
    clipping_threshold: float = 0.01,
) -> AudioQualityStatus:
    """Summarize signal quality; quality labels do not authenticate a speaker."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    finite = np.nan_to_num(samples, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size == 0:
        return AudioQualityStatus(
            "invalid", 0.0, 0.0, 0.0, dropped_audio_chunks, dropped_audio_samples, skipped_inference_windows,
            ("empty audio window",),
        )
    peak = float(np.max(np.abs(finite)))
    rms = float(np.sqrt(np.mean(np.square(finite, dtype=np.float64))))
    clipped_fraction = float(np.mean(np.abs(finite) >= 0.999))
    details: list[str] = []
    if rms < silence_rms_threshold:
        details.append("silence_or_very_low_level")
    if clipped_fraction >= clipping_threshold:
        details.append("clipping_detected")
    if dropped_audio_chunks or dropped_audio_samples:
        details.append("microphone_input_overflow_detected")
    if skipped_inference_windows:
        details.append("inference_worker_skipped_stale_windows")
    if not details:
        details.append("signal_level_check_passed")
    status = "ok" if details == ["signal_level_check_passed"] else "+".join(details)
    return AudioQualityStatus(
        status,
        rms,
        peak,
        clipped_fraction,
        dropped_audio_chunks,
        dropped_audio_samples,
        skipped_inference_windows,
        tuple(details),
    )


class LocalVoicePredictor:
    """Load this project's own checkpoint and run local LFCC CNN inference.

    The audio/features config is read from the checkpoint, not from a separate
    mutable file, so live preprocessing matches the selected training run.
    Checkpoints must be trusted files created by this project's trainer.
    """

    def __init__(self, checkpoint_path: Path | str, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = self._resolve_device(device)
        if not self.checkpoint_path.is_file():
            raise ModelLoadError(f"Checkpoint not found: {self.checkpoint_path}")
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
            raise ModelLoadError(
                f"Could not load checkpoint {self.checkpoint_path}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(checkpoint, dict):
            raise ModelLoadError("Checkpoint must be a project training checkpoint dictionary")
        if checkpoint.get("class_mapping") != {"bona_fide": 0, "spoof": 1}:
            raise ModelLoadError("Checkpoint class mapping must be bona_fide=0, spoof=1")
        metadata = checkpoint.get("training_metadata")
        if not isinstance(metadata, dict) or metadata.get("evaluation_split_used") is not False:
            raise ModelLoadError("Checkpoint is not marked as produced without evaluation-split selection")
        if checkpoint.get("architecture_settings", {}).get("name") != "LFCCCNNDetector":
            raise ModelLoadError("Checkpoint architecture is not this project's LFCCCNNDetector")
        raw_config = checkpoint.get("preprocessing_config")
        if not isinstance(raw_config, dict) or not isinstance(raw_config.get("audio"), dict) or not isinstance(
            raw_config.get("features"), dict
        ):
            raise ModelLoadError("Checkpoint does not include the exact audio/features config")
        try:
            self.preprocessing = self._preprocessing_from_checkpoint(raw_config)
            settings = checkpoint["architecture_settings"]
            self.model = LFCCCNNDetector(
                dropout=float(settings["dropout"]),
                seed=int(checkpoint.get("random_seed", 42)),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model.to(self.device).eval()
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ModelLoadError(f"Checkpoint model/config is incompatible: {exc}") from exc
        self.checkpoint_epoch = int(checkpoint.get("epoch", 0))
        self.checkpoint_metadata = metadata

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA unavailable; using CPU for local inference.", flush=True)
            return torch.device("cpu")
        try:
            return torch.device(device)
        except RuntimeError as exc:
            raise ModelLoadError(f"Unsupported inference device {device!r}") from exc

    @staticmethod
    def _preprocessing_from_checkpoint(raw_config: dict[str, Any]) -> PreprocessingConfig:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="voice_antispoof_config_") as temp_dir:
            path = Path(temp_dir) / "checkpoint_config.yaml"
            path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
            return load_preprocessing_config(path)

    def predict_window(
        self,
        waveform_16khz_mono: np.ndarray,
        *,
        dropped_audio_chunks: int = 0,
        dropped_audio_samples: int = 0,
        skipped_inference_windows: int = 0,
    ) -> dict[str, Any]:
        """Return class scores, signal quality, and model-only latency."""
        waveform = np.asarray(waveform_16khz_mono, dtype=np.float32).reshape(-1)
        if waveform.size != self.preprocessing.samples_per_window:
            raise ValueError(
                f"Expected exactly {self.preprocessing.samples_per_window} samples at "
                f"{self.preprocessing.audio.sample_rate_hz} Hz; got {waveform.size}"
            )
        quality = assess_audio_quality(
            waveform,
            dropped_audio_chunks=dropped_audio_chunks,
            dropped_audio_samples=dropped_audio_samples,
            skipped_inference_windows=skipped_inference_windows,
        )
        _, features = process_waveform(
            waveform,
            self.preprocessing.audio.sample_rate_hz,
            self.preprocessing,
        )
        inputs = torch.from_numpy(np.array(features[:, None, ...], dtype=np.float32, copy=True)).to(
            self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(inputs)
            scores = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "bona_fide_score": float(scores[0]),
            "spoof_score": float(scores[1]),
            "audio_quality": quality.to_dict(),
            "inference_latency_ms": latency_ms,
            "device": str(self.device),
            "checkpoint_epoch": self.checkpoint_epoch,
            "interpretation": "Model estimate only; not verified identity or proof of spoofing.",
        }
