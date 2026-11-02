"""Predictor tests use a generated checkpoint and synthetic waveforms only."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from inference.predict import LocalVoicePredictor, ModelLoadError, assess_audio_quality
from models.cnn_detector import LFCCCNNDetector
from streaming.microphone import LiveInferenceResult, LiveMicrophoneSession

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"


def _write_own_checkpoint(path: Path) -> None:
    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    model = LFCCCNNDetector(dropout=0.3, seed=42)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture_settings": {"name": "LFCCCNNDetector", "dropout": 0.3},
            "random_seed": 42,
            "class_mapping": {"bona_fide": 0, "spoof": 1},
            "preprocessing_config": config,
            "training_metadata": {"evaluation_split_used": False},
            "epoch": 3,
        },
        path,
    )


def test_local_predictor_returns_estimates_and_quality(tmp_path: Path) -> None:
    checkpoint = tmp_path / "own-checkpoint.pt"
    _write_own_checkpoint(checkpoint)
    predictor = LocalVoicePredictor(checkpoint, device="cpu")
    waveform = 0.2 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000).astype(np.float32)
    result = predictor.predict_window(waveform)
    assert 0.0 <= result["bona_fide_score"] <= 1.0
    assert 0.0 <= result["spoof_score"] <= 1.0
    assert result["bona_fide_score"] + result["spoof_score"] == pytest.approx(1.0)
    assert result["inference_latency_ms"] >= 0
    assert result["audio_quality"]["status"] == "ok"
    assert "not verified identity" in result["interpretation"]


def test_silence_clipping_and_dropped_chunks_are_reported() -> None:
    silence = assess_audio_quality(np.zeros(32000, dtype=np.float32))
    assert "silence" in silence.status
    clipped = assess_audio_quality(
        np.ones(32000, dtype=np.float32), dropped_audio_chunks=1, skipped_inference_windows=2
    )
    assert "clipping" in clipped.status
    assert "overflow" in clipped.status
    assert "skipped" in clipped.status


def test_microphone_callback_only_buffers_worker_runs_inference() -> None:
    """Fake callback blocks exercise threading without opening a real mic."""
    caller_thread = threading.current_thread().name
    inference_threads: list[str] = []

    class FakePredictor:
        preprocessing = SimpleNamespace(
            audio=SimpleNamespace(
                sample_rate_hz=16000,
                fixed_window_seconds=2.0,
                window_hop_seconds=0.5,
            )
        )

        @staticmethod
        def predict_window(waveform, **kwargs):
            inference_threads.append(threading.current_thread().name)
            return {
                "bona_fide_score": 0.6,
                "spoof_score": 0.4,
                "audio_quality": {"status": "ok"},
                "inference_latency_ms": 1.0,
                "device": "cpu",
                "checkpoint_epoch": 1,
                "interpretation": "Model estimate only; not verified identity.",
            }

    session = LiveMicrophoneSession(FakePredictor(), blocksize=1600)
    worker = threading.Thread(target=session._worker_loop, name="test-inference-worker", daemon=True)
    session._worker = worker
    worker.start()
    for _ in range(20):
        session._audio_callback(np.full((1600, 1), 0.1, dtype=np.float32), 1600, None, None)
    assert session.results.empty(), "callback must not run or publish inference results itself"
    result = session.get_result(timeout=10)
    session.stop()

    assert isinstance(result, LiveInferenceResult)
    assert inference_threads == ["test-inference-worker"]
    assert caller_thread != inference_threads[0]
    assert result.bona_fide_score == 0.6


def test_bad_checkpoint_and_wrong_window_fail_clearly(tmp_path: Path) -> None:
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save({"arbitrary": "data"}, bad_checkpoint)
    with pytest.raises(ModelLoadError):
        LocalVoicePredictor(bad_checkpoint, device="cpu")

    checkpoint = tmp_path / "own-checkpoint.pt"
    _write_own_checkpoint(checkpoint)
    predictor = LocalVoicePredictor(checkpoint, device="cpu")
    with pytest.raises(ValueError, match="Expected exactly"):
        predictor.predict_window(np.zeros(100, dtype=np.float32))
