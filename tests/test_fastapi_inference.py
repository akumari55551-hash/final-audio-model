"""FastAPI inference handler tests using the real local checkpoint for audio runs."""
from __future__ import annotations

import io
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi import UploadFile
from starlette.middleware.cors import CORSMiddleware

from backend.app.main import app, _checkpoint_path, analyze, health
from inference.predict import LocalVoicePredictor
from inference.predict import ModelLoadError


def wav_bytes() -> bytes:
    samples = (np.sin(np.arange(16000 * 2, dtype=np.float32) * 2 * np.pi * 220 / 16000) * 0.1)
    pcm = (samples * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    app.state.predictor = LocalVoicePredictor(_checkpoint_path())
    app.state.checkpoint_sha256 = __import__("hashlib").sha256(_checkpoint_path().read_bytes()).hexdigest()
    app.state.model_error = None
    app.state.predictor_lock = __import__("threading").Lock()
    return True


def upload(name: str, content: bytes, mime: str = "audio/wav") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content), headers={"content-type": mime})


def test_health_reports_loaded_checkpoint(client):
    body = health()
    assert body["status"] == "online"
    assert body["model_loaded"] is True
    assert body["model"]["name"] == "LFCCCNNDetector"
    assert len(body["model"]["checkpoint_sha256"]) == 64


def test_active_vite_origin_is_allowed_for_browser_api_calls():
    cors = next(middleware for middleware in app.user_middleware if middleware.cls is CORSMiddleware)
    assert "http://127.0.0.1:5174" in cors.kwargs["allow_origins"]


def test_missing_checkpoint_is_reported_as_model_load_error(tmp_path: Path):
    with pytest.raises(ModelLoadError, match="Checkpoint not found"):
        LocalVoicePredictor(tmp_path / "missing.pt")


def test_invalid_audio_is_rejected(client):
    with pytest.raises(Exception) as error:
        analyze(upload("bad.wav", b"not audio"))
    assert getattr(error.value, "status_code", None) == 422


def test_oversized_upload_is_rejected(client):
    with pytest.raises(Exception) as error:
        analyze(upload("large.wav", b"0" * (50 * 1024 * 1024 + 1)))
    assert getattr(error.value, "status_code", None) == 413


def test_model_unavailable_returns_503(client):
    predictor = app.state.predictor
    app.state.predictor = None
    try:
        with pytest.raises(Exception) as error:
            analyze(upload("tone.wav", wav_bytes()))
        assert getattr(error.value, "status_code", None) == 503
    finally:
        app.state.predictor = predictor


def test_health_distinguishes_backend_online_and_model_unavailable(client):
    predictor, error = app.state.predictor, app.state.model_error
    app.state.predictor = None
    app.state.model_error = "Model unavailable (ModelLoadError). Configure a valid SONARA_CHECKPOINT_PATH."
    try:
        body = health()
        assert body["status"] == "online"
        assert body["model_loaded"] is False
        assert body["model"] is None
        assert "Model unavailable" in body["error"]
    finally:
        app.state.predictor, app.state.model_error = predictor, error


def test_predictor_exception_returns_safe_inference_error(client, tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes())
    predictor = app.state.predictor
    original = predictor.predict_window
    predictor.predict_window = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("internal detail"))
    try:
        with pytest.raises(Exception) as error:
            analyze(upload("tone.wav", source.read_bytes()))
        assert getattr(error.value, "status_code", None) == 500
        assert "internal detail" not in str(error.value)
    finally:
        predictor.predict_window = original


@pytest.mark.parametrize("format_name,content_type", [("wav", "audio/wav"), ("m4a", "audio/mp4")])
def test_real_inference_for_wav_and_m4a(client, tmp_path: Path, format_name: str, content_type: str):
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes())
    target = tmp_path / f"tone.{format_name}"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-c:a", "aac", "-b:a", "96k", str(target)] if format_name == "m4a" else ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-c:a", "pcm_s16le", str(target)]
    subprocess.run(command, check=True)
    body = analyze(upload(target.name, target.read_bytes(), content_type))
    assert body["classification"] in {"BONA_FIDE", "SPOOF", "UNKNOWN"}
    assert 0 <= body["bona_fide_score"] <= 1
    assert 0 <= body["spoof_score"] <= 1
    assert body["detection"]["status"] == "AVAILABLE"
    assert body["provenance"]["class_mapping"] == {"bona_fide": 0, "spoof": 1}
    assert body["provenance"]["score_aggregation"] == "softmax_mean_window_logits"
    assert body["provenance"]["inference_id"] == body["request_id"]
    window_predictions = body["window_predictions"]
    assert window_predictions
    assert [item["window_index"] for item in window_predictions] == list(range(len(window_predictions)))
    assert all(len(item["logits"]) == 2 for item in window_predictions)
    expected_scores = torch.softmax(torch.tensor([item["logits"] for item in window_predictions]).mean(dim=0), dim=0)
    assert body["bona_fide_score"] == pytest.approx(expected_scores[0].item(), abs=1e-7)
    assert body["spoof_score"] == pytest.approx(expected_scores[1].item(), abs=1e-7)
    expected_class = ("BONA_FIDE", "SPOOF")[int(expected_scores.argmax().item())]
    assert body["classification"] == expected_class
    assert all(item["predicted_class"] == ("BONA_FIDE" if item["logits"][0] > item["logits"][1] else "SPOOF" if item["logits"][1] > item["logits"][0] else "UNKNOWN") for item in window_predictions)
    assert body["audio_metadata"]["duration_seconds"] == pytest.approx(2, abs=0.03)
    assert body["audio_metadata"]["prepared_sample_rate_hz"] == 16000
    assert body["audio_metadata"]["prepared_channels"] == 1
    assert body["audio_metadata"]["prepared_encoding"] == "pcm_s16le"
    assert body["audio_metadata"]["sha256_original"] == __import__("hashlib").sha256(target.read_bytes()).hexdigest()
