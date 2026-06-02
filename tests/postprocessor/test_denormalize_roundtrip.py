"""Inverse-property tests for `normalize_spectrogram` ↔ `denormalize_spectrogram`.

PR #71 covered the forward direction in `tests/preprocessor/test_spectral_extended.py`.
These tests pin the inverse property: normalizing then denormalizing must recover
the input within floating-point tolerance, for every supported method.
"""

import numpy as np
import pytest

from stemmy.postprocessing.spectral import denormalize_spectrogram
from stemmy.preprocessing.spectral import normalize_spectrogram


@pytest.fixture
def magnitude_2d():
    rng = np.random.default_rng(0)
    # Non-negative magnitudes with a realistic dynamic range.
    return rng.uniform(0.0, 10.0, size=(64, 128))


@pytest.fixture
def magnitude_stereo():
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 10.0, size=(2, 64, 128))


def test_minmax_roundtrip(magnitude_2d):
    normalized, params = normalize_spectrogram(magnitude_2d, method="minmax")
    restored = denormalize_spectrogram(normalized, params)
    assert restored.shape == magnitude_2d.shape
    assert np.allclose(restored, magnitude_2d, atol=1e-6)


def test_freq_minmax_roundtrip(magnitude_2d):
    normalized, params = normalize_spectrogram(magnitude_2d, method="freq_minmax")
    restored = denormalize_spectrogram(normalized, params)
    assert restored.shape == magnitude_2d.shape
    assert np.allclose(restored, magnitude_2d, atol=1e-6)


def test_freq_minmax_roundtrip_stereo(magnitude_stereo):
    """Stereo (3D) magnitudes must roundtrip too — freq stats are per-frequency, per-channel."""
    normalized, params = normalize_spectrogram(magnitude_stereo, method="freq_minmax")
    restored = denormalize_spectrogram(normalized, params)
    assert restored.shape == magnitude_stereo.shape
    assert np.allclose(restored, magnitude_stereo, atol=1e-6)


def test_none_method_is_pure_identity():
    """method='none' returns a copy on normalize and a copy on denormalize — no transformation."""
    mag = np.array([[1.0, 2.0], [3.0, 4.0]])
    normalized, params = normalize_spectrogram(mag, method="none")
    restored = denormalize_spectrogram(normalized, params)
    assert np.array_equal(restored, mag)
    # `none` must not silently share storage; both directions copy.
    assert restored is not mag


def test_freq_minmax_handles_constant_frequency_row():
    """If one frequency bin is constant across time (f_max == f_min), restored value
    must equal the original constant — no NaN, no Inf."""
    mag = np.ones((4, 8), dtype=np.float64) * 3.7  # entirely constant
    normalized, params = normalize_spectrogram(mag, method="freq_minmax")
    restored = denormalize_spectrogram(normalized, params)
    assert np.isfinite(restored).all()
    assert np.allclose(restored, mag, atol=1e-6)


def test_minmax_constant_input_roundtrips_to_constant():
    """If the entire spectrogram has a single value, roundtrip must preserve it."""
    mag = np.full((16, 16), 0.5, dtype=np.float64)
    normalized, params = normalize_spectrogram(mag, method="minmax")
    restored = denormalize_spectrogram(normalized, params)
    assert np.isfinite(restored).all()
    assert np.allclose(restored, mag, atol=1e-6)


def test_denormalize_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        denormalize_spectrogram(np.zeros((4, 4)), {"method": "log"})
