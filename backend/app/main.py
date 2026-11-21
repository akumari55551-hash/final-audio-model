"""A.U.D.I.O local FastAPI inference service."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference.predict import LocalVoicePredictor
from training.audio_preprocessing import AudioProcessingError, process_audio_file

ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
FORMATS = {".m4a", ".wav", ".wave", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
ALLOWED_CONTAINERS = {
    ".m4a": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".wav": {"wav"}, ".wave": {"wav"}, ".flac": {"flac"},
    ".mp3": {"mp3"}, ".ogg": {"ogg"}, ".aiff": {"aiff", "aif"}, ".aif": {"aiff", "aif"},
}
MIME_BY_SUFFIX = {
    ".m4a": {"audio/mp4", "audio/x-m4a", "audio/m4a", "application/octet-stream"},
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"},
    ".wave": {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"},
    ".flac": {"audio/flac", "audio/x-flac", "application/octet-stream"},
    ".mp3": {"audio/mpeg", "audio/mp3", "application/octet-stream"},
    ".ogg": {"audio/ogg", "application/ogg", "application/octet-stream"},
    ".aiff": {"audio/aiff", "audio/x-aiff", "application/octet-stream"},
    ".aif": {"audio/aiff", "audio/x-aiff", "application/octet-stream"},
}


def _checkpoint_path() -> Path:
    configured = os.getenv("SONARA_CHECKPOINT_PATH")
    return Path(configured).expanduser() if configured else ROOT / "checkpoints" / "user_audio_human_plus_ai" / "best.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.model_error = None
    app.state.predictor_lock = threading.Lock()
    checkpoint = _checkpoint_path()
    try:
        app.state.predictor = LocalVoicePredictor(checkpoint, os.getenv("SONARA_DEVICE", "auto"))
        app.state.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    except Exception as exc:  # service remains healthy so callers can distinguish API/model status
        app.state.model_error = f"Model unavailable ({type(exc).__name__}). Configure a valid SONARA_CHECKPOINT_PATH."
        app.state.checkpoint_sha256 = None
    yield


app = FastAPI(title="A.U.D.I.O Inference API", version="1.0.0", lifespan=lifespan)
origins = [item.strip() for item in os.getenv("SONARA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict[str, Any]:
    predictor = app.state.predictor
    model = None
    if predictor:
        model = {
            "name": "LFCCCNNDetector",
            "version": f"checkpoint-epoch-{predictor.checkpoint_epoch}",
            "checkpoint_sha256": app.state.checkpoint_sha256,
        }
    return {
        "status": "online", "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else None,
        "model": model, "error": app.state.model_error,
    }


def _run_process(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Audio decoding failed. Ensure FFmpeg is installed and the audio codec is supported.") from exc
    if result.returncode:
        raise ValueError("Audio is corrupted or uses an unsupported codec.")
    return result


def _analyze_bytes(data: bytes, filename: str, content_type: str) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in FORMATS:
        raise ValueError("Unsupported audio format. Use M4A, WAV, FLAC, MP3, OGG, or AIFF.")
    mime = (content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if mime not in MIME_BY_SUFFIX[suffix]:
        raise ValueError("File extension and audio MIME type do not match.")
    if not data:
        raise ValueError("Audio file is empty.")
    ffmpeg = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
    ffprobe = os.getenv("FFPROBE_BINARY") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("FFmpeg and ffprobe are required to decode uploaded audio.")
    with tempfile.TemporaryDirectory(prefix="sonara-") as directory:
        source = Path(directory) / f"original{suffix}"
        converted = Path(directory) / "prepared.wav"
        source.write_bytes(data)
        probe = _run_process([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "format=format_name,duration:stream=sample_rate,channels", "-of", "json", str(source)])
        try:
            metadata = json.loads(probe.stdout)
            fmt = set(metadata["format"]["format_name"].split(","))
            stream = metadata["streams"][0]
            duration = float(metadata["format"]["duration"])
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("The file does not contain a readable audio stream.") from exc
        if not fmt.intersection(ALLOWED_CONTAINERS[suffix]):
            raise ValueError("File extension does not match its detected audio content.")
        if duration <= 0 or duration > 60 * 60:
            raise ValueError("Audio duration must be greater than zero and no more than one hour.")
        _run_process([ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(converted)])
        # Decode the normalized WAV through the same configured preprocessing used by training.
        # The live predictor owns the exact checkpoint preprocessing config.
        predictor = app.state.predictor
        windows, _ = process_audio_file(converted, predictor.preprocessing)
        predictions = [predictor.predict_window(window) for window in windows]
        mean_logits = np.mean(np.asarray([item["logits"] for item in predictions], dtype=np.float32), axis=0)
        recording_scores = torch.softmax(torch.from_numpy(mean_logits), dim=0).tolist()
        bona_fide, spoof = (float(score) for score in recording_scores)
        if abs(bona_fide - spoof) < 1e-12:
            classification = "UNKNOWN"
        else:
            classification = "BONA_FIDE" if bona_fide > spoof else "SPOOF"
        request_id = str(uuid.uuid4())
        provenance = {
            "model_name": "LFCCCNNDetector", "model_version": f"checkpoint-epoch-{predictor.checkpoint_epoch}",
            "checkpoint_sha256": app.state.checkpoint_sha256, "inference_id": request_id,
            "generated_at": datetime.now(timezone.utc).isoformat(), "source": "local_fastapi_checkpoint",
            "device": str(predictor.device),
            "class_mapping": {"bona_fide": 0, "spoof": 1},
            "score_aggregation": "softmax_mean_window_logits", "evaluation_status": "independent_test_pending",
        }
        return {
            "request_id": request_id, "status": "completed", "classification": classification,
            "bona_fide_score": bona_fide, "spoof_score": spoof, "model_status": "available",
            "model_version": provenance["model_version"], "processing_time_ms": sum(x["inference_latency_ms"] for x in predictions),
            "audio_metadata": {"filename": Path(filename).name, "original_format": suffix.lstrip(".").upper(), "file_size_bytes": len(data), "sha256_original": hashlib.sha256(data).hexdigest(), "duration_seconds": duration, "sample_rate_hz": sample_rate, "channel_count": channels, "prepared_sample_rate_hz": 16000, "prepared_channels": 1, "prepared_encoding": "pcm_s16le"},
            "provenance": provenance,
            "window_predictions": [
                {
                    "window_index": index,
                    "logits": item["logits"],
                    "bona_fide_score": item["bona_fide_score"],
                    "spoof_score": item["spoof_score"],
                    "predicted_class": "BONA_FIDE" if item["logits"][0] > item["logits"][1] else "SPOOF" if item["logits"][1] > item["logits"][0] else "UNKNOWN",
                }
                for index, item in enumerate(predictions)
            ],
            "detection": {"status": "AVAILABLE", "authenticity": classification, "confidence": max(bona_fide, spoof), "bona_fide_score": bona_fide, "spoof_score": spoof, "provenance": provenance, "explanation": "Experimental model estimate only; not verified identity or proof of spoofing."},
        }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Inference model is unavailable. Check /api/health for model status.")
    chunks: list[bytes] = []
    total = 0
    while chunk := file.file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio exceeds the 50 MB upload limit.")
        chunks.append(chunk)
    data = b"".join(chunks)
    start = time.perf_counter()
    try:
        with app.state.predictor_lock:
            result = _analyze_bytes(data, file.filename or "audio", file.content_type or "")
        result["processing_time_ms"] = max(result["processing_time_ms"], (time.perf_counter() - start) * 1000)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (AudioProcessingError, sf.SoundFileRuntimeError) as exc:
        raise HTTPException(status_code=422, detail="Audio preprocessing failed. Verify the file is a supported, valid recording.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed ({type(exc).__name__}).") from exc
