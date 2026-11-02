"""Inspect preprocessing tensor shapes for one locally downloaded dataset file.

Example (after obtaining official ASVspoof 2019 LA data):

    python -m training.examples.inspect_lfcc `
      data/raw/ASVspoof2019_LA_train/flac/LA_T_XXXXXXXX.flac

Replace the placeholder filename with an actual file present on your machine.
This example does not download data, create fixtures, or train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.audio_preprocessing import load_preprocessing_config, process_audio_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path, help="Path to a real, locally available audio file")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Shared preprocessing YAML (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    config = load_preprocessing_config(args.config)
    waveform_windows, lfcc_features = process_audio_file(args.audio_file, config)
    waveform_tensor = torch.from_numpy(waveform_windows)
    lfcc_tensor = torch.from_numpy(lfcc_features)

    print(f"Input file: {args.audio_file.resolve()}")
    print(f"Configured sample rate: {config.audio.sample_rate_hz} Hz, mono")
    print(f"Waveform windows tensor shape: {tuple(waveform_tensor.shape)}")
    print(f"LFCC tensor shape: {tuple(lfcc_tensor.shape)}")
    print(f"LFCC finite: {bool(torch.isfinite(lfcc_tensor).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
