"""Small custom LFCC 2D CNN for two-class bona-fide/spoof logits.

All parameters are initialized locally from random distributions; this module
has no checkpoint loading, pretrained architecture, or inference API behavior.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.lfcc import DEFAULT_LFCC_INPUT_SPEC, LFCCInputSpec


class _ConvBlock(nn.Sequential):
    """Two padded convolutions followed by batch normalization and pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class LFCCCNNDetector(nn.Module):
    """Randomly initialized CNN mapping LFCC maps to bona-fide/spoof logits.

    Input shape is ``[batch, 1, n_lfcc, frames]``. The three convolutional
    blocks use 16, 32, and 64 channels, each with two 3x3 convolutions,
    batch normalization, ReLU, and 2x2 max pooling. Adaptive global average
    pooling removes dependence on frame count; dropout (p=0.3) regularizes a
    64-to-2 linear classification head. Outputs are raw logits ordered as
    ``(bona_fide, spoof)``; apply softmax only in downstream evaluation code.

    A minimum of 8 LFCC bins and 8 frames is required by the three pooling
    stages. The default ``seed`` makes initialization reproducible without
    changing the caller's global RNG state. Pass ``seed=None`` to use the
    current PyTorch RNG state.
    """

    num_classes = 2
    class_names = ("bona_fide", "spoof")

    def __init__(
        self,
        *,
        dropout: float = 0.3,
        seed: int | None = 42,
        input_spec: LFCCInputSpec = DEFAULT_LFCC_INPUT_SPEC,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative or None")
        self.input_spec = input_spec
        # Layer constructors themselves consume random values, so both module
        # construction and explicit initialization belong inside the private
        # RNG scope. This preserves the caller's global CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.features = nn.Sequential(
                _ConvBlock(1, 16),
                _ConvBlock(16, 32),
                _ConvBlock(32, 64),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(64, self.num_classes)
            )
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every trainable layer locally; no pretrained weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # Randomize trainable scale parameters rather than copying a
                # learned/pretrained normalization state.
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return unnormalized class logits with shape ``[batch, 2]``."""
        self.input_spec.validate(inputs)
        features = self.features(inputs)
        return self.classifier(features)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by gradient-based training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
