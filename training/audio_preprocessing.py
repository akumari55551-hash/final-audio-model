"""Shared deterministic waveform preprocessing and LFCC extraction.

Training and inference must load the same ``configs/baseline.yaml`` through
:func:`load_preprocessing_config` and call :func:`process_audio_file`. No
normalization statistics are estimated or fitted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.fft import dct
from scipy.signal import resample_poly
from scipy.signal.windows import hann

SUPPORTED_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".aiff", ".aif"}


class AudioProcessingError(ValueError):
    """Raised for empty, unsupported, or undecodable audio inputs."""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    fixed_window_seconds: float = 2.0
    window_hop_seconds: float = 0.5
    amplitude_policy: str = "sanitize_and_clip"
    resampler: str = "scipy_resample_poly"
    resampler_window: str = "hann"


@dataclass(frozen=True)
class LFCCConfig:
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_filters: int = 64
    n_coefficients: int = 40
    f_min_hz: float = 0.0
    f_max_hz: float = 8000.0
    preemphasis: float = 0.97
    log_energy_floor: float = 1.0e-10
    spectrum: str = "power"
    window: str = "hann_periodic"
    center: bool = False
    normalization: str = "none"


@dataclass(frozen=True)
class PreprocessingConfig:
    audio: AudioConfig = AudioConfig()
    features: LFCCConfig = LFCCConfig()

    @property
    def samples_per_window(self) -> int:
        return round(self.audio.sample_rate_hz * self.audio.fixed_window_seconds)

    @property
    def window_hop_samples(self) -> int:
        return round(self.audio.sample_rate_hz * self.audio.window_hop_seconds)

    @property
    def frame_count(self) -> int:
        # center=False; a complete STFT frame is required.
        return 1 + (self.samples_per_window - self.features.win_length) // self.features.hop_length


def _validate_config(config: PreprocessingConfig) -> None:
    audio = config.audio
    features = config.features
    if audio.sample_rate_hz <= 0 or audio.channels != 1:
        raise ValueError("Preprocessing requires a positive target rate and mono output (channels=1)")
    if audio.fixed_window_seconds <= 0 or audio.window_hop_seconds <= 0:
        raise ValueError("Window duration and hop must be positive")
    if audio.amplitude_policy != "sanitize_and_clip":
        raise ValueError("Only the explicit sanitize_and_clip amplitude policy is supported")
    if audio.resampler != "scipy_resample_poly" or audio.resampler_window != "hann":
        raise ValueError("Only scipy resample_poly with the configured Hann window is supported")
    if features.center:
        raise ValueError("center=True is not supported; framing is explicitly non-centered")
    if features.normalization != "none":
        raise ValueError("Only normalization='none' is supported; do not fit on evaluation data")
    if features.spectrum != "power" or features.window != "hann_periodic":
        raise ValueError("LFCC requires power spectrum and hann_periodic window in this implementation")
    if features.n_fft <= 0 or features.win_length <= 0 or features.hop_length <= 0:
        raise ValueError("FFT, frame length, and frame hop must be positive")
    if features.win_length > features.n_fft:
        raise ValueError("win_length cannot exceed n_fft")
    if features.n_filters < 2 or not 1 <= features.n_coefficients <= features.n_filters:
        raise ValueError("Require n_filters >= 2 and 1 <= n_coefficients <= n_filters")
    if not 0 <= features.f_min_hz < features.f_max_hz <= audio.sample_rate_hz / 2:
        raise ValueError("LFCC frequency bounds must lie within [0, Nyquist]")
    if not 0.0 <= features.preemphasis < 1.0:
        raise ValueError("preemphasis must be in [0, 1)")
    if features.log_energy_floor <= 0:
        raise ValueError("log_energy_floor must be positive")
    if config.samples_per_window < features.win_length:
        raise ValueError("Window duration is too short for one LFCC frame")


def load_preprocessing_config(config_path: Path | str) -> PreprocessingConfig:
    """Load and validate the shared YAML audio/features sections."""
    path = Path(config_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load preprocessing config {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("audio"), dict) or not isinstance(
        raw.get("features"), dict
    ):
        raise TypeError(f"{path} must contain mapping sections 'audio' and 'features'")
    audio_raw: dict[str, Any] = raw["audio"]
    features_raw: dict[str, Any] = raw["features"]
    if features_raw.get("name") != "lfcc":
        raise ValueError("features.name must be 'lfcc'")

    audio_config = AudioConfig(
        sample_rate_hz=int(audio_raw.get("sample_rate_hz", 16000)),
        channels=int(audio_raw.get("channels", 1)),
        fixed_window_seconds=float(
            audio_raw.get("fixed_window_seconds", audio_raw.get("window_seconds", 2.0))
        ),
        window_hop_seconds=float(
            audio_raw.get("window_hop_seconds", audio_raw.get("hop_seconds", 0.5))
        ),
        amplitude_policy=str(audio_raw.get("amplitude_policy", "sanitize_and_clip")),
        resampler=str(audio_raw.get("resampler", "scipy_resample_poly")),
        resampler_window=str(audio_raw.get("resampler_window", "hann")),
    )
    feature_config = LFCCConfig(
        n_fft=int(features_raw.get("n_fft", 512)),
        win_length=int(features_raw.get("win_length", 400)),
        hop_length=int(features_raw.get("hop_length", 160)),
        n_filters=int(features_raw.get("n_filters", 64)),
        n_coefficients=int(features_raw.get("n_coefficients", 40)),
        f_min_hz=float(features_raw.get("f_min_hz", 0.0)),
        f_max_hz=float(features_raw.get("f_max_hz", 8000.0)),
        preemphasis=float(features_raw.get("preemphasis", 0.97)),
        log_energy_floor=float(features_raw.get("log_energy_floor", 1.0e-10)),
        spectrum=str(features_raw.get("spectrum", "power")),
        window=str(features_raw.get("window", "hann_periodic")),
        center=bool(features_raw.get("center", False)),
        normalization=str(features_raw.get("normalization", "none")),
    )
    config = PreprocessingConfig(audio=audio_config, features=feature_config)
    _validate_config(config)
    return config


def _coerce_mono_float32(waveform: np.ndarray) -> np.ndarray:
    """Convert samples-by-channels or mono data to sanitized, clipped mono."""
    samples = np.asarray(waveform)
    if samples.size == 0:
        raise AudioProcessingError("Audio contains no samples")
    if samples.ndim == 2:
        if samples.shape[1] == 0:
            raise AudioProcessingError("Audio contains no channels")
        channels = np.asarray(samples, dtype=np.float32)
        channels = np.nan_to_num(channels, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        samples = np.mean(channels.astype(np.float64, copy=False), axis=1)
    elif samples.ndim != 1:
        raise AudioProcessingError("Audio must be a mono vector or samples-by-channels matrix")
    if samples.size == 0:
        raise AudioProcessingError("Audio contains no samples")
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.nan_to_num(samples, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(samples, -1.0, 1.0, out=samples)
    return np.ascontiguousarray(samples, dtype=np.float32)


def preprocess_waveform(
    waveform: np.ndarray,
    source_sample_rate_hz: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Sanitize, downmix, and deterministically resample a waveform to mono 16 kHz.

    Multi-channel audio is averaged across channels. NaN and infinities become
    zero; finite amplitudes are clipped to [-1, 1] without per-file peak
    normalization. ``scipy.signal.resample_poly`` performs rational polyphase
    resampling with a fixed Hann filter and constant boundary extension.
    """
    _validate_config(config)
    if source_sample_rate_hz <= 0:
        raise AudioProcessingError("Source sample rate must be positive")
    mono = _coerce_mono_float32(waveform)
    target_rate = config.audio.sample_rate_hz
    if source_sample_rate_hz == target_rate:
        return mono

    divisor = math.gcd(source_sample_rate_hz, target_rate)
    up = target_rate // divisor
    down = source_sample_rate_hz // divisor
    resampled = resample_poly(mono, up, down, window="hann", padtype="constant")
    return _coerce_mono_float32(resampled)


def load_audio_file(path: Path | str, config: PreprocessingConfig) -> np.ndarray:
    """Load supported WAV/FLAC/OGG/AIFF audio, downmix, sanitize, and resample."""
    audio_path = Path(path)
    if audio_path.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise AudioProcessingError(
            f"Unsupported audio extension {audio_path.suffix!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if not audio_path.is_file():
        raise AudioProcessingError(f"Audio file does not exist: {audio_path}")
    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.SoundFileRuntimeError, ValueError) as exc:
        raise AudioProcessingError(f"Cannot decode audio file {audio_path}: {exc}") from exc
    return preprocess_waveform(waveform, int(sample_rate), config)


def make_fixed_windows(
    waveform: np.ndarray, config: PreprocessingConfig
) -> np.ndarray:
    """Return [window, sample] float32 windows of fixed length.

    Short waveforms are right-zero-padded to one window. Longer waveforms use
    left-aligned windows at the configured hop. A final partial tail is
    right-zero-padded; no samples are silently dropped. Exact-boundary endings
    do not create an extra fully padded window.
    """
    _validate_config(config)
    samples = _coerce_mono_float32(waveform)
    window_size = config.samples_per_window
    hop = config.window_hop_samples
    if hop <= 0:
        raise ValueError("Window hop resolves to fewer than one sample")

    if samples.size <= window_size:
        starts = [0]
    else:
        starts = list(range(0, samples.size - window_size + 1, hop))
        last_start = starts[-1]
        if last_start + window_size < samples.size:
            starts.append(last_start + hop)

    windows = np.zeros((len(starts), window_size), dtype=np.float32)
    for index, start in enumerate(starts):
        end = min(start + window_size, samples.size)
        windows[index, : end - start] = samples[start:end]
    return windows


def _linear_filterbank(config: LFCCConfig, sample_rate_hz: int) -> np.ndarray:
    """Create triangular filters evenly spaced in linear frequency (Hz)."""
    frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / sample_rate_hz)
    edges = np.linspace(config.f_min_hz, config.f_max_hz, config.n_filters + 2)
    filters = np.zeros((config.n_filters, frequencies.size), dtype=np.float64)
    for filter_index in range(config.n_filters):
        left, center, right = edges[filter_index : filter_index + 3]
        rising = (frequencies - left) / (center - left)
        falling = (right - frequencies) / (right - center)
        filters[filter_index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def extract_lfcc(windows: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Extract log-frequency cepstral coefficients from fixed waveform windows.

    Parameters are taken exclusively from ``configs/baseline.yaml``. Each
    frame is pre-emphasized, multiplied by a periodic Hann window, zero-padded
    to ``n_fft``, transformed by an RFFT, converted to power, integrated in
    linear-Hz triangular filters, log-compressed with a floor, and transformed
    with an orthonormal type-II DCT. Coefficients 0 through N-1 are retained.

    Returns a finite float32 array shaped ``[window, coefficient, frame]``.
    Framing is left-aligned and non-centered; trailing incomplete frames are
    omitted (the configured fixed windows are longer than one frame).
    """
    _validate_config(config)
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise AudioProcessingError("windows must have shape [samples] or [window, samples]")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise AudioProcessingError("Cannot extract LFCC from empty windows")
    if values.shape[1] != config.samples_per_window:
        raise AudioProcessingError(
            f"Each window must have exactly {config.samples_per_window} samples; got {values.shape[1]}"
        )
    values = np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(values, -1.0, 1.0, out=values)

    audio_config = config.audio
    feature_config = config.features
    frame_count = config.frame_count
    frame_starts = np.arange(frame_count, dtype=np.int64) * feature_config.hop_length
    frame_offsets = np.arange(feature_config.win_length, dtype=np.int64)
    frames = values[:, frame_starts[:, None] + frame_offsets[None, :]]

    if feature_config.preemphasis:
        emphasized = np.empty_like(frames)
        emphasized[..., 0] = frames[..., 0]
        emphasized[..., 1:] = frames[..., 1:] - feature_config.preemphasis * frames[..., :-1]
        frames = emphasized

    window = hann(feature_config.win_length, sym=False).astype(np.float32)
    frames = frames * window
    spectrum = np.fft.rfft(frames, n=feature_config.n_fft, axis=-1)
    power = np.square(np.abs(spectrum), dtype=np.float64)
    filterbank = _linear_filterbank(feature_config, audio_config.sample_rate_hz)
    energies = np.matmul(power, filterbank.T)
    log_energies = np.log(np.maximum(energies, feature_config.log_energy_floor))
    coefficients = dct(log_energies, type=2, norm="ortho", axis=-1)
    coefficients = coefficients[..., : feature_config.n_coefficients]
    coefficients = np.transpose(coefficients, (0, 2, 1))
    result = np.asarray(coefficients, dtype=np.float32)
    if not np.isfinite(result).all():
        raise AudioProcessingError("LFCC extraction produced non-finite values")
    return result


def process_waveform(
    waveform: np.ndarray,
    source_sample_rate_hz: int,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared end-to-end in-memory path: sanitize/resample, window, LFCC."""
    prepared = preprocess_waveform(waveform, source_sample_rate_hz, config)
    windows = make_fixed_windows(prepared, config)
    return windows, extract_lfcc(windows, config)


def process_audio_file(
    path: Path | str, config: PreprocessingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Shared end-to-end file path used by both training and inference."""
    waveform = load_audio_file(path, config)
    windows = make_fixed_windows(waveform, config)
    return windows, extract_lfcc(windows, config)
