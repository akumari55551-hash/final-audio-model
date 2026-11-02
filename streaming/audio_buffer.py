"""Thread-safe bounded mono audio ring buffer for overlapping live windows."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioWindow:
    """A copied audio window and its ending sample index."""

    waveform: np.ndarray
    end_sample: int
    skipped_windows_before: int = 0


class BoundedAudioBuffer:
    """Fixed-memory circular buffer emitting fixed-size windows at a hop.

    ``append`` is designed for the PortAudio callback: it performs bounded
    copying and bookkeeping only, never feature extraction or model inference.
    The consumer calls ``pop_ready_windows`` from a worker thread. On a late
    consumer, stale windows that have already been overwritten are skipped and
    counted instead of returning corrupted audio.
    """

    def __init__(self, sample_rate_hz: int = 16000, window_seconds: float = 2.0, hop_seconds: float = 0.5):
        if sample_rate_hz <= 0 or window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("sample rate, window duration, and hop must be positive")
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = round(sample_rate_hz * window_seconds)
        self.hop_samples = round(sample_rate_hz * hop_seconds)
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("window and hop must resolve to at least one sample")
        self.capacity_samples = self.window_samples
        self._ring = np.zeros(self.capacity_samples, dtype=np.float32)
        self._write_index = 0
        self._total_samples = 0
        self._available_start = 0
        self._next_window_end = self.window_samples
        self._lock = threading.Lock()
        self.dropped_input_samples = 0
        self.dropped_windows = 0
        self.discontinuities = 0

    @property
    def buffered_samples(self) -> int:
        with self._lock:
            return min(self._total_samples - self._available_start, self.capacity_samples)

    def reset_after_discontinuity(self) -> None:
        """Discard partial history after a sounddevice input-overflow event."""
        with self._lock:
            self.discontinuities += 1
            self._available_start = self._total_samples
            self._next_window_end = self._total_samples + self.window_samples

    def append(self, samples: np.ndarray) -> None:
        """Append a mono callback block, clipping/sanitizing into fixed storage."""
        block = np.asarray(samples, dtype=np.float32)
        if block.ndim == 2:
            if block.shape[1] == 0:
                raise ValueError("audio block has no channels")
            block = np.mean(block, axis=1, dtype=np.float64).astype(np.float32)
        elif block.ndim != 1:
            raise ValueError("audio block must have shape [frames] or [frames, channels]")
        if block.size == 0:
            return
        block = np.nan_to_num(block, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(block, -1.0, 1.0, out=block)

        with self._lock:
            if block.size > self.capacity_samples:
                original_size = int(block.size)
                skipped = original_size - self.capacity_samples
                self.dropped_input_samples += skipped
                block = block[-self.capacity_samples :]
                self._available_start = self._total_samples + skipped
                self._write_index = (self._write_index + skipped) % self.capacity_samples
                self._total_samples += skipped

            first = min(block.size, self.capacity_samples - self._write_index)
            self._ring[self._write_index : self._write_index + first] = block[:first]
            remaining = block.size - first
            if remaining:
                self._ring[:remaining] = block[first:]
            self._write_index = (self._write_index + block.size) % self.capacity_samples
            self._total_samples += int(block.size)
            self._available_start = max(self._available_start, self._total_samples - self.capacity_samples)

    def _copy_window_ending_at(self, end_sample: int) -> np.ndarray:
        start_sample = end_sample - self.window_samples
        if start_sample < self._available_start:
            raise RuntimeError("requested audio window was overwritten before consumption")
        end_index = end_sample % self.capacity_samples
        start_index = (end_index - self.window_samples) % self.capacity_samples
        if start_index < end_index:
            return self._ring[start_index:end_index].copy()
        if start_index == end_index:
            # The window exactly spans the ring capacity.
            return np.concatenate((self._ring[start_index:], self._ring[:end_index])).astype(
                np.float32, copy=False
            )
        return np.concatenate((self._ring[start_index:], self._ring[:end_index])).astype(
            np.float32, copy=False
        )

    def pop_ready_windows(self) -> list[AudioWindow]:
        """Return every due, still-buffered window; called by a worker thread."""
        with self._lock:
            if self._total_samples < self._next_window_end:
                return []
            windows: list[AudioWindow] = []
            skipped_since_last = 0
            earliest_valid_end = self._available_start + self.window_samples
            if self._next_window_end < earliest_valid_end:
                skipped = math.ceil((earliest_valid_end - self._next_window_end) / self.hop_samples)
                self._next_window_end += skipped * self.hop_samples
                self.dropped_windows += skipped
                skipped_since_last = skipped
            while self._next_window_end <= self._total_samples:
                end = self._next_window_end
                waveform = self._copy_window_ending_at(end)
                windows.append(
                    AudioWindow(
                        waveform=waveform,
                        end_sample=end,
                        skipped_windows_before=skipped_since_last,
                    )
                )
                skipped_since_last = 0
                self._next_window_end += self.hop_samples
            return windows
