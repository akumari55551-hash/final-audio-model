"""Architecture tests; random tensors are shape checks, not performance evidence."""

from __future__ import annotations

import pytest
import torch

from models.cnn_detector import LFCCCNNDetector, count_trainable_parameters
from models.lfcc import set_reproducible_seed


def test_cpu_forward_shape_finite_logits_and_parameter_count() -> None:
    model = LFCCCNNDetector(seed=42).cpu().eval()
    inputs = torch.randn(3, 1, 40, 198, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(inputs)

    parameter_count = count_trainable_parameters(model)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert parameter_count > 0
    assert parameter_count == sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 1_000_000


def test_variable_frame_lengths_are_supported() -> None:
    model = LFCCCNNDetector(seed=7).eval()
    for frames in (8, 13, 101, 198, 257):
        with torch.inference_mode():
            logits = model(torch.zeros(2, 1, 40, frames))
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()


def test_same_seed_reproduces_initial_weights() -> None:
    first = LFCCCNNDetector(seed=1234)
    second = LFCCCNNDetector(seed=1234)
    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_constructor_seed_does_not_mutate_global_cpu_rng() -> None:
    torch.manual_seed(2025)
    expected_next = torch.rand(1)
    torch.manual_seed(2025)
    _ = LFCCCNNDetector(seed=1)
    actual_next = torch.rand(1)
    assert torch.equal(actual_next, expected_next)


def test_reproducible_seed_helper_seeds_cpu_rng() -> None:
    set_reproducible_seed(77)
    first = torch.rand(4)
    set_reproducible_seed(77)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_input_shape_and_dtype_errors_are_clear() -> None:
    model = LFCCCNNDetector(seed=1)
    with pytest.raises(ValueError, match=r"shape \[batch, 1, n_lfcc, frames\]"):
        model(torch.zeros(1, 40, 198))
    with pytest.raises(ValueError, match="at least 8 frames"):
        model(torch.zeros(1, 1, 40, 7))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(1, 1, 40, 16, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_forward_and_parameter_device() -> None:
    model = LFCCCNNDetector(seed=42).cuda().eval()
    inputs = torch.randn(2, 1, 40, 128, device="cuda")
    with torch.inference_mode():
        logits = model(inputs)
    torch.cuda.synchronize()

    assert logits.shape == (2, 2)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()
