"""Input contract and reproducibility helpers for LFCC-based models."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class LFCCInputSpec:
    """Expected LFCC tensor layout: batch, channel, coefficient, frame."""

    channels: int = 1
    minimum_coefficients: int = 8
    minimum_frames: int = 8

    def validate(self, value: Tensor) -> None:
        """Raise a clear error unless value meets the CNN input contract."""
        if value.ndim != 4:
            raise ValueError(
                "LFCC input must have shape [batch, 1, n_lfcc, frames]; "
                f"received {tuple(value.shape)}"
            )
        batch, channels, coefficients, frames = value.shape
        if batch < 1:
            raise ValueError("LFCC input batch dimension must be at least 1")
        if channels != self.channels:
            raise ValueError(f"LFCC input channel dimension must be {self.channels}, got {channels}")
        if coefficients < self.minimum_coefficients:
            raise ValueError(
                f"LFCC input needs at least {self.minimum_coefficients} coefficients, got {coefficients}"
            )
        if frames < self.minimum_frames:
            raise ValueError(
                f"LFCC input needs at least {self.minimum_frames} frames, got {frames}"
            )
        if not value.is_floating_point():
            raise TypeError(f"LFCC input must have floating-point dtype, got {value.dtype}")


def set_reproducible_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for repeatable experiments.

    Deterministic algorithms are optional because some operations can be slower
    or unavailable on particular CUDA/PyTorch combinations. This helper never
    downloads or loads pretrained weights.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


DEFAULT_LFCC_INPUT_SPEC = LFCCInputSpec()
