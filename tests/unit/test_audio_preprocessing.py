"""Synthetic unit fixtures only; no benchmark dataset records are embedded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.audio_preprocessing import (
    AudioProcessingError,
    load_audio_file,
    load_preprocessing_config,
    make_fixed_windows,
    preprocess_waveform,
    process_audio_file,
    process_waveform,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "baseline.yaml"


@pytest.fixture(scope="module")
def config():
    return load_preprocessing_config(CONFIG_PATH)


def test_configured_feature_dimensions_are_fixed(config) -> None:
    samples = np.zeros(config.samples_per_window, dtype=np.float32)
    windows, features = process_waveform(samples, config.audio.sample_rate_hz, config)
    assert windows.shape == (1, config.samples_per_window)
    assert features.shape == (1, config.features.n_coefficients, config.frame_count)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_short_audio_is_right_padded_to_one_window(config) -> None:
    short = np.ones(320, dtype=np.float32) * 0.25
    windows = make_fixed_windows(short, config)
    assert windows.shape == (1, config.samples_per_window)
    assert np.allclose(windows[0, : short.size], short)
    assert np.count_nonzero(windows[0, short.size :]) == 0


def test_long_audio_uses_deterministic_overlapping_windows(config) -> None:
    duration_samples = config.samples_per_window + 2 * config.window_hop_samples + 100
    waveform = np.linspace(-0.5, 0.5, duration_samples, dtype=np.float32)
    first = make_fixed_windows(waveform, config)
    second = make_fixed_windows(waveform, config)
    assert first.shape == (4, config.samples_per_window)
    np.testing.assert_array_equal(first, second)


def test_resampling_and_lfcc_are_deterministic(config) -> None:
    time = np.arange(44100, dtype=np.float32) / 44100
    signal = 0.2 * np.sin(2 * np.pi * 440 * time)
    windows_a, features_a = process_waveform(signal, 44100, config)
    windows_b, features_b = process_waveform(signal, 44100, config)
    np.testing.assert_array_equal(windows_a, windows_b)
    np.testing.assert_array_equal(features_a, features_b)
    assert windows_a.shape[1] == config.samples_per_window


def test_empty_audio_raises_a_clear_error(config) -> None:
    with pytest.raises(AudioProcessingError, match="no samples"):
        preprocess_waveform(np.array([], dtype=np.float32), 16000, config)


def test_nan_inf_clipping_and_stereo_downmix_are_safe(config) -> None:
    stereo = np.array(
        [[np.nan, 2.0], [np.inf, -2.0], [-np.inf, 0.5], [0.2, 0.4]],
        dtype=np.float32,
    )
    processed = preprocess_waveform(stereo, 16000, config)
    assert processed.ndim == 1
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 1.0


def test_silence_produces_finite_features(config) -> None:
    windows, features = process_waveform(np.zeros(1600, dtype=np.float32), 16000, config)
    assert windows.shape == (1, config.samples_per_window)
    assert np.isfinite(features).all()


def test_unsupported_and_corrupt_audio_files_are_rejected(tmp_path: Path, config) -> None:
    unsupported = tmp_path / "recording.mp3"
    unsupported.write_bytes(b"not actually an MP3")
    with pytest.raises(AudioProcessingError, match="Unsupported audio extension"):
        load_audio_file(unsupported, config)

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a valid WAV")
    with pytest.raises(AudioProcessingError, match="Cannot decode audio file"):
        load_audio_file(corrupt, config)


def test_audio_file_path_matches_waveform_pipeline(tmp_path: Path, config) -> None:
    """A temporary WAV verifies file loading only; it is not benchmark audio."""
    path = tmp_path / "synthetic_fixture.wav"
    waveform = np.linspace(-0.3, 0.3, 8000, dtype=np.float32)
    sf.write(path, waveform, 16000, subtype="FLOAT")
    file_windows, file_features = process_audio_file(path, config)
    array_windows, array_features = process_waveform(waveform, 16000, config)
    np.testing.assert_allclose(file_windows, array_windows, atol=1e-7)
    np.testing.assert_allclose(file_features, array_features, atol=1e-6)
