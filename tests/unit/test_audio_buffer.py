"""Ring-buffer tests use generated sample arrays, not microphone recordings."""

from __future__ import annotations

import numpy as np
import pytest

from streaming.audio_buffer import BoundedAudioBuffer


def test_initial_window_and_half_second_hop() -> None:
    buffer = BoundedAudioBuffer(sample_rate_hz=10, window_seconds=2.0, hop_seconds=0.5)
    buffer.append(np.arange(20, dtype=np.float32) / 100)
    windows = buffer.pop_ready_windows()
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0].waveform, np.arange(20, dtype=np.float32) / 100)
    assert windows[0].end_sample == 20

    buffer.append(np.arange(20, 25, dtype=np.float32) / 100)
    next_window = buffer.pop_ready_windows()
    assert len(next_window) == 1
    np.testing.assert_array_equal(next_window[0].waveform, np.arange(5, 25, dtype=np.float32) / 100)
    assert next_window[0].end_sample == 25


def test_late_worker_skips_overwritten_windows_without_corrupt_data() -> None:
    buffer = BoundedAudioBuffer(sample_rate_hz=10, window_seconds=2.0, hop_seconds=0.5)
    buffer.append(np.arange(50, dtype=np.float32) / 100)
    windows = buffer.pop_ready_windows()
    assert windows
    assert all(window.waveform.size == 20 for window in windows)
    assert buffer.dropped_windows > 0
    for window in windows:
        np.testing.assert_array_equal(
            window.waveform,
            np.arange(window.end_sample - 20, window.end_sample, dtype=np.float32) / 100,
        )


def test_oversized_block_is_bounded_and_recent_window_is_aligned() -> None:
    buffer = BoundedAudioBuffer(sample_rate_hz=10, window_seconds=2.0, hop_seconds=0.5)
    buffer.append(np.arange(25, dtype=np.float32) / 100)
    assert buffer.buffered_samples == 20
    assert buffer.dropped_input_samples == 5
    windows = buffer.pop_ready_windows()
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0].waveform, np.arange(5, 25, dtype=np.float32) / 100)


def test_discontinuity_discards_partial_window_history() -> None:
    buffer = BoundedAudioBuffer(sample_rate_hz=10, window_seconds=2.0, hop_seconds=0.5)
    buffer.append(np.ones(15, dtype=np.float32))
    buffer.reset_after_discontinuity()
    assert buffer.pop_ready_windows() == []
    buffer.append(np.arange(20, dtype=np.float32) / 100)
    windows = buffer.pop_ready_windows()
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0].waveform, np.arange(20, dtype=np.float32) / 100)
    assert buffer.discontinuities == 1


def test_stereo_downmix_clips_and_sanitizes() -> None:
    buffer = BoundedAudioBuffer(sample_rate_hz=4, window_seconds=1, hop_seconds=0.5)
    buffer.append(np.array([[np.nan, 2.0], [0.5, 0.5], [np.inf, 0.0], [-2.0, -2.0]], dtype=np.float32))
    windows = buffer.pop_ready_windows()
    assert len(windows) == 1
    assert np.isfinite(windows[0].waveform).all()
    assert float(np.max(np.abs(windows[0].waveform))) <= 1.0


def test_invalid_buffer_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        BoundedAudioBuffer(sample_rate_hz=0)
