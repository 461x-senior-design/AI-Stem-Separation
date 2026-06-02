"""Tests for time-chunked model execution in `_run_model_forward`.

The chunking path is the mitigation for OOM on long tracks. These tests
verify that chunked inference is mathematically equivalent (within float
tolerance) to a single-pass forward, so the mitigation produces the same
output the model would produce if memory were unlimited.

Also covers `_validate_chunk_settings` error branches.
"""

import pytest
import torch

from stemmy.inference import (
    InferenceConfig,
    InferenceException,
    _run_model_forward,
    _validate_chunk_settings,
)
from stemmy.models.unet_2d import UNet2D

# A small UNet keeps the test fast. UNet2D.forward returns 5D [B, S, C, F, T]; the
# rank-generic chunking code at inference.py:669 handles this correctly. Note: there
# is a separate downstream mismatch at inference.py:786 that asserts the model output
# is 4D — see Trello `Bug:` ticket for the model-output rank inconsistency.
SMALL_UNET_KWARGS = dict(stems=4, audio_channels=1, base_channels=8)

# F=64 (multiple of 16 so no padding is needed), T=192 (divides cleanly into 128+64).
INPUT_SHAPE = (1, 1, 64, 192)


@pytest.fixture
def model():
    m = UNet2D(**SMALL_UNET_KWARGS)
    m.eval()
    return m


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(*INPUT_SHAPE)


@pytest.fixture
def cpu():
    return torch.device("cpu")


def test_chunked_equals_unchunked_with_overlap(model, x, cpu):
    """Overlap-add chunking must produce output very close to a single-pass forward.

    Residual boundary effects from conv padding + cross-platform BLAS variance
    (MKL on Linux vs Accelerate on macOS) push the worst single-cell difference
    into the ~1e-3 to 5e-3 range. We assert two bounds:
      * mean diff stays << 1e-3 — chunking is not systemically biased
      * max diff stays < 1e-2 — no single cell diverges catastrophically
    """
    cfg_full = InferenceConfig(chunk_frames=0)
    cfg_chunk = InferenceConfig(chunk_frames=128, overlap_frames=32)

    ref = _run_model_forward(model, x, cpu, cfg_full)
    chunked = _run_model_forward(model, x, cpu, cfg_chunk)

    assert ref.shape == chunked.shape
    assert ref.shape == (1, 4, 1, 64, 192)

    diff = (ref - chunked).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    assert mean_diff < 1e-3, f"mean abs diff too large: {mean_diff:.2e}"
    assert max_diff < 1e-2, f"max abs diff too large: {max_diff:.2e}"


def test_chunked_no_overlap_runs_cleanly(model, x, cpu):
    """No-overlap chunking has expected boundary artifacts, but must produce valid output.

    This test verifies the OOM-mitigation path runs end-to-end even without overlap:
    correct shape, no NaN/Inf, mean deviation from reference stays small.
    """
    cfg_full = InferenceConfig(chunk_frames=0)
    cfg_chunk = InferenceConfig(chunk_frames=96, overlap_frames=0)

    ref = _run_model_forward(model, x, cpu, cfg_full)
    chunked = _run_model_forward(model, x, cpu, cfg_chunk)

    assert ref.shape == chunked.shape
    assert torch.isfinite(chunked).all()
    # Boundary artifacts are bounded; mean-abs diff stays well below 1e-2.
    assert (ref - chunked).abs().mean().item() < 1e-2


@pytest.mark.parametrize(
    "chunk_frames,overlap_frames",
    [
        (192, 0),  # chunk == full length, single chunk
        (128, 32),  # standard overlap
        (96, 16),  # smaller chunk, smaller overlap
        (64, 8),  # many small chunks
    ],
)
def test_chunked_output_shape_invariant(model, x, cpu, chunk_frames, overlap_frames):
    """Output time dimension must equal input time dimension across (chunk, overlap) combos."""
    cfg = InferenceConfig(chunk_frames=chunk_frames, overlap_frames=overlap_frames)
    out = _run_model_forward(model, x, cpu, cfg)

    assert out.shape == (1, 4, 1, 64, 192)


def test_chunked_overflow_chunk_clamps_to_length(model, x, cpu):
    """If chunk_frames > t_frames, validation clamps to t_frames and the model runs once."""
    cfg = InferenceConfig(chunk_frames=10_000, overlap_frames=0)
    out = _run_model_forward(model, x, cpu, cfg)
    assert out.shape == (1, 4, 1, 64, 192)


def test_validate_chunk_settings_zero_chunk_disables_chunking():
    """chunk_frames <= 0 is the disabled-chunking sentinel; returns (0, 0)."""
    cfg = InferenceConfig(chunk_frames=0, overlap_frames=0)
    assert _validate_chunk_settings(cfg, t_frames=100) == (0, 0)

    cfg_neg = InferenceConfig(chunk_frames=-5, overlap_frames=0)
    assert _validate_chunk_settings(cfg_neg, t_frames=100) == (0, 0)


def test_validate_chunk_settings_negative_overlap_raises():
    cfg = InferenceConfig(chunk_frames=128, overlap_frames=-1)
    with pytest.raises(InferenceException, match="overlap_frames must be >= 0"):
        _validate_chunk_settings(cfg, t_frames=512)


def test_validate_chunk_settings_overlap_not_less_than_chunk_raises():
    cfg = InferenceConfig(chunk_frames=128, overlap_frames=128)
    with pytest.raises(InferenceException, match="overlap_frames must be < chunk_frames"):
        _validate_chunk_settings(cfg, t_frames=512)

    cfg2 = InferenceConfig(chunk_frames=128, overlap_frames=200)
    with pytest.raises(InferenceException, match="overlap_frames must be < chunk_frames"):
        _validate_chunk_settings(cfg2, t_frames=512)


def test_validate_chunk_settings_zero_t_frames_raises_when_chunking_enabled():
    cfg = InferenceConfig(chunk_frames=64, overlap_frames=0)
    with pytest.raises(InferenceException, match="Invalid input tensor time dimension"):
        _validate_chunk_settings(cfg, t_frames=0)


def test_validate_chunk_settings_chunk_larger_than_t_clamps():
    """chunk_frames > t_frames is allowed and clamped (no exception)."""
    cfg = InferenceConfig(chunk_frames=10_000, overlap_frames=0)
    chunk, overlap = _validate_chunk_settings(cfg, t_frames=192)
    assert chunk == 192
    assert overlap == 0
