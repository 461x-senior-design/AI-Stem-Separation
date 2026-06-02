"""White-box tests for `_renorm_sum_to_one`.

The renorm step ensures soft-mask values across all stems sum to 1 at each
time-frequency cell, so reconstructed stems sum back to the input mixture.
These tests pin the invariants: shape preservation, sum-to-one, NaN safety
on zero inputs, and identity on already-normalized inputs.
"""

import pytest
import torch

from stemmy.inference import _renorm_sum_to_one


def test_renorm_sums_to_one_across_stems():
    """Output must sum to 1.0 along the stem dimension at every (F, T) cell."""
    torch.manual_seed(0)
    raw = torch.rand(1, 4, 32, 64)  # softmax-like, all positive

    out = _renorm_sum_to_one(raw)

    sums = out.sum(dim=1)
    assert sums.shape == (1, 32, 64)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_renorm_preserves_shape():
    raw = torch.rand(2, 4, 16, 8)
    out = _renorm_sum_to_one(raw)
    assert out.shape == (2, 4, 16, 8)


def test_renorm_zero_input_is_finite():
    """All-zero input must not produce NaN/Inf — the eps clamp guards division."""
    zeros = torch.zeros(1, 4, 8, 8)

    out = _renorm_sum_to_one(zeros)

    assert torch.isfinite(out).all()
    # With eps clamp, all-zero numerator / eps denominator = 0 across all stems.
    assert torch.allclose(out, torch.zeros_like(out))


def test_renorm_already_normalized_is_identity():
    """Input that already sums to 1 along stems must remain (almost) unchanged."""
    # Build an explicitly sum-to-one input via softmax.
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 16, 16)
    normalized = torch.softmax(logits, dim=1)

    assert torch.allclose(normalized.sum(dim=1), torch.ones(1, 16, 16), atol=1e-6)

    out = _renorm_sum_to_one(normalized)

    assert torch.allclose(out, normalized, atol=1e-6)


def test_renorm_preserves_relative_proportions():
    """Renorm is a per-cell scaling; the ratio between any two stems must be preserved."""
    raw = torch.tensor(
        [
            [
                [[2.0]],  # stem 0
                [[1.0]],  # stem 1
                [[1.0]],  # stem 2
                [[0.0]],  # stem 3
            ]
        ]
    )

    out = _renorm_sum_to_one(raw)

    assert torch.allclose(out.sum(dim=1), torch.ones(1, 1, 1), atol=1e-6)
    # Stem 0 was 2x stem 1 in the raw values; should remain 2x after rescaling.
    assert torch.isclose(out[0, 0, 0, 0], 2.0 * out[0, 1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("eps", [1e-8, 1e-6, 1e-4])
def test_renorm_eps_does_not_corrupt_normal_case(eps):
    """A reasonable range of eps values should not visibly change well-behaved outputs."""
    torch.manual_seed(2)
    raw = torch.rand(1, 4, 8, 8) + 0.1  # well above eps

    out = _renorm_sum_to_one(raw, eps=eps)

    assert torch.allclose(out.sum(dim=1), torch.ones(1, 8, 8), atol=1e-6)
