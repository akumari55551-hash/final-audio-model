"""Sounddevice microphone capture with worker-thread local inference.

The PortAudio callback only appends a small block to a bounded ring and signals
an event. Preprocessing, CNN inference, JSON formatting, and console output all
run on the worker thread, outside the realtime callback.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from inference.predict import LocalVoicePredictor, ModelLoadError
from streaming.audio_buffer import BoundedAudioBuffer


class MicrophoneUnavailableError(RuntimeError):
    """Raised when the microphone cannot be enumerated or opened."""


@dataclass(frozen=True)
class LiveInferenceResult:
    sequence: int
    timestamp_utc: str
    bona_fide_score: float
    spoof_score: float
    audio_quality: dict[str, Any]
    inference_latency_ms: float
    device: str
    checkpoint_epoch: int
    interpretation: str
    window_end_sample: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "bona_fide_score": self.bona_fide_score,
            "spoof_score": self.spoof_score,
            "audio_quality": self.audio_quality,
            "inference_latency_ms": self.inference_latency_ms,
            "device": self.device,
            "checkpoint_epoch": self.checkpoint_epoch,
            "interpretation": self.interpretation,
            "window_end_sample": self.window_end_sample,
        }


class LiveMicrophoneSession:
    """Own the microphone stream, bounded audio ring, worker, and result queue."""

    def __init__(
        self,
        predictor: LocalVoicePredictor,
        *,
        device: int | str | None = None,
        blocksize: int = 1600,
        result_queue_size: int = 8,
    ) -> None:
        if blocksize < 1 or result_queue_size < 1:
            raise ValueError("blocksize and result_queue_size must be positive")
        self.predictor = predictor
        self.sample_rate_hz = predictor.preprocessing.audio.sample_rate_hz
        self.buffer = BoundedAudioBuffer(
            sample_rate_hz=self.sample_rate_hz,
            window_seconds=predictor.preprocessing.audio.fixed_window_seconds,
            hop_seconds=predictor.preprocessing.audio.window_hop_seconds,
        )
        self.device = device
        self.blocksize = blocksize
        self.results: queue.Queue[LiveInferenceResult | dict[str, Any]] = queue.Queue(
            maxsize=result_queue_size
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._worker_error: BaseException | None = None
        self._callback_error: BaseException | None = None
        self._input_overflows = 0
        self._dropped_result_messages = 0
        self._sequence = 0
        self._stream = None
        self._worker: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._worker is not None and self._worker.is_alive()

    @staticmethod
    def _sounddevice_module():
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MicrophoneUnavailableError(
                "sounddevice is not installed. Activate the project .venv and run "
                "python -m pip install -r requirements.txt."
            ) from exc
        return sd

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Realtime callback: bounded append and signal only; no inference."""
        del frames, time_info
        try:
            if status:
                status_text = str(status).casefold()
                if "overflow" in status_text:
                    self._input_overflows += 1
                    self.buffer.reset_after_discontinuity()
            self.buffer.append(indata)
            self._ready.set()
        except (ValueError, RuntimeError, TypeError, FloatingPointError) as exc:
            # Store and hand off errors; do not print or infer in callback.
            self._callback_error = exc
            self._stop.set()
            self._ready.set()

    def _publish(self, result: LiveInferenceResult | dict[str, Any]) -> None:
        try:
            self.results.put_nowait(result)
        except queue.Full:
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self._dropped_result_messages += 1
            try:
                self.results.put_nowait(result)
            except queue.Full:
                self._dropped_result_messages += 1

    def _worker_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._ready.wait(timeout=0.25)
                self._ready.clear()
                if self._callback_error is not None:
                    raise RuntimeError(f"Audio callback failed: {self._callback_error}") from self._callback_error
                windows = self.buffer.pop_ready_windows()
                for window in windows:
                    if self._stop.is_set():
                        break
                    payload = self.predictor.predict_window(
                        window.waveform,
                        dropped_audio_chunks=self._input_overflows,
                        dropped_audio_samples=self.buffer.dropped_input_samples,
                        skipped_inference_windows=window.skipped_windows_before,
                    )
                    self._sequence += 1
                    self._publish(
                        LiveInferenceResult(
                            sequence=self._sequence,
                            timestamp_utc=datetime.now(UTC).isoformat(),
                            bona_fide_score=payload["bona_fide_score"],
                            spoof_score=payload["spoof_score"],
                            audio_quality=payload["audio_quality"],
                            inference_latency_ms=payload["inference_latency_ms"],
                            device=payload["device"],
                            checkpoint_epoch=payload["checkpoint_epoch"],
                            interpretation=payload["interpretation"],
                            window_end_sample=window.end_sample,
                        )
                    )
        except (OSError, ValueError, RuntimeError, TypeError, FloatingPointError) as exc:
            self._worker_error = exc
            self._publish({"error": f"Local inference worker stopped: {type(exc).__name__}: {exc}"})
            self._stop.set()

    def start(self) -> None:
        """Open a mono 16 kHz stream and launch the non-realtime worker."""
        if self._worker is not None:
            raise RuntimeError("Microphone session has already been started")
        sd = self._sounddevice_module()
        try:
            device_info = sd.query_devices(self.device, "input")
            if int(device_info["max_input_channels"]) < 1:
                raise MicrophoneUnavailableError("Selected device has no input channels")
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="voice-antispoof-inference",
                daemon=True,
            )
            self._worker.start()
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate_hz,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                callback=self._audio_callback,
            )
            self._stream.start()
        except MicrophoneUnavailableError:
            self.stop()
            raise
        except Exception as exc:
            self.stop()
            raise MicrophoneUnavailableError(
                f"Could not open a mono {self.sample_rate_hz} Hz microphone stream: "
                f"{type(exc).__name__}: {exc}. Check Windows microphone privacy permissions, "
                "device selection, and whether another application has exclusive access."
            ) from exc

    def get_result(self, timeout: float | None = None) -> LiveInferenceResult | dict[str, Any]:
        """Wait for one prediction/error message and surface worker failures."""
        if self._worker_error is not None and self.results.empty():
            raise RuntimeError(f"Inference worker failed: {self._worker_error}") from self._worker_error
        return self.results.get(timeout=timeout)

    def stop(self) -> None:
        """Stop stream and worker safely; no capture is retained to disk."""
        self._stop.set()
        self._ready.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except (RuntimeError, OSError):
                self._worker_error = self._worker_error or RuntimeError(
                    "sounddevice stream stop reported an error"
                )
            try:
                self._stream.close()
            except (RuntimeError, OSError):
                self._worker_error = self._worker_error or RuntimeError(
                    "sounddevice stream close reported an error"
                )
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                self._worker_error = RuntimeError("Inference worker did not stop within 10 seconds")
            self._worker = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def _format_result(result: LiveInferenceResult | dict[str, Any]) -> str:
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    quality = result.audio_quality
    return (
        f"window={result.sequence} bona_fide={result.bona_fide_score:.4f} "
        f"spoof={result.spoof_score:.4f} quality={quality['status']} "
        f"inference={result.inference_latency_ms:.2f}ms device={result.device} | "
        f"MODEL ESTIMATE ONLY — not verified identity"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Your project's selected best.pt")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--microphone-device", type=int, help="sounddevice input device index; default input otherwise")
    parser.add_argument("--blocksize", type=int, default=1600, help="Audio callback frames; default 100 ms at 16 kHz")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        try:
            sd = LiveMicrophoneSession._sounddevice_module()
            print(sd.query_devices())
            return 0
        except MicrophoneUnavailableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    try:
        predictor = LocalVoicePredictor(args.checkpoint, device=args.device)
        print(
            f"Loaded local checkpoint epoch {predictor.checkpoint_epoch} on {predictor.device}. "
            "Audio stays in memory. Press Ctrl+C to stop.",
            flush=True,
        )
        with LiveMicrophoneSession(
            predictor,
            device=args.microphone_device,
            blocksize=args.blocksize,
        ) as session:
            print(
                f"Capturing mono {predictor.preprocessing.audio.sample_rate_hz} Hz; "
                f"{predictor.preprocessing.audio.fixed_window_seconds:.1f}s windows, "
                f"updates every {predictor.preprocessing.audio.window_hop_seconds:.3f}s.",
                flush=True,
            )
            while True:
                try:
                    result = session.get_result(timeout=1.0)
                except queue.Empty:
                    continue
                print(_format_result(result), flush=True)
    except KeyboardInterrupt:
        print("\nMicrophone session stopped.", flush=True)
        return 0
    except (MicrophoneUnavailableError, ModelLoadError, RuntimeError, ValueError, queue.Empty) as exc:
        print(f"ERROR: live inference stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
