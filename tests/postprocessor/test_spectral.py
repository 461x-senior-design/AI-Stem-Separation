import numpy as np
import pytest

from stemmy.postprocessing.spectral import (
    apply_mask,
    combine_magnitude_phase,
    compute_istft,
    denormalize_spectrogram,
)
from stemmy.preprocessing.spectral import compute_stft, normalize_spectrogram


def test_apply_mask_basic():
    """Test that mask is applied correctly via element-wise multiplication."""
    # Arrange
    magnitude = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[0.5, 1.0], [0.0, 0.5]])

    # Act
    result = apply_mask(magnitude, mask)

    # Assert
    expected = np.array([[0.5, 2.0], [0.0, 2.0]])
    np.testing.assert_array_almost_equal(result, expected)


def test_apply_mask_ones_preserves_magnitude():
    """Test that mask of all ones returns original magnitude."""
    # Arrange
    rng = np.random.default_rng(0)
    magnitude = rng.random((2, 100, 50))

    mask = np.ones_like(magnitude)

    # Act
    result = apply_mask(magnitude, mask)

    # Assert
    np.testing.assert_array_equal(result, magnitude)


def test_apply_mask_zeros_silences():
    """Test that mask of all zeros returns zeros."""
    # Arrange
    rng = np.random.default_rng(1)
    magnitude = rng.random((2, 100, 50))
    mask = np.zeros_like(magnitude)

    # Act
    result = apply_mask(magnitude, mask)

    # Assert
    assert np.all(result == 0)


def test_combine_magnitude_phase_basic():
    """Test that magnitude and phase combine correctly."""
    # Arrange - known values
    rng = np.random.default_rng(2)
    magnitude = rng.random((2, 100, 50))
    phase = rng.uniform(-np.pi, np.pi, (2, 100, 50))

    # Act
    result = combine_magnitude_phase(magnitude, phase)

    # Assert
    # mag=2, phase=0 -> 2+0j
    assert result[0] == pytest.approx(2.0 + 0j)
    # mag=1, phase=pi/2 -> 0+1j (approximately)
    assert result[1].real == pytest.approx(0.0, abs=1e-10)
    assert result[1].imag == pytest.approx(1.0)
    # mag=3, phase=pi -> -3+0j
    assert result[2].real == pytest.approx(-3.0)
    assert result[2].imag == pytest.approx(0.0, abs=1e-10)


def test_combine_magnitude_phase_output_is_complex():
    """Test that output is complex-valued."""
    # Arrange
    rng = np.random.default_rng(3)
    magnitude = rng.random((2, 100, 50))
    phase = rng.uniform(-np.pi, np.pi, (2, 100, 50))

    # Act
    result = combine_magnitude_phase(magnitude, phase)

    # Assert
    assert np.iscomplexobj(result)


def test_combine_magnitude_phase_preserves_magnitude():
    """Test that |combine(mag, phase)| == mag."""
    # Arrange
    rng = np.random.default_rng(4)
    original_stft = rng.random((2, 100, 50)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, (2, 100, 50))
    )

    # Act
    result = combine_magnitude_phase(magnitude, phase)

    # Assert - magnitude of result should equal original magnitude
    np.testing.assert_array_almost_equal(np.abs(result), magnitude)


def test_combine_is_inverse_of_split():
    """Test that combine reverses split_magnitude_phase."""
    # Arrange - create a complex STFT
    from stemmy.preprocessing.spectral import split_magnitude_phase

    original_stft = np.random.rand(2, 100, 50) * np.exp(
        1j * np.random.uniform(-np.pi, np.pi, (2, 100, 50))
    )

    # Act - split then recombine
    magnitude, phase = split_magnitude_phase(original_stft)
    reconstructed = combine_magnitude_phase(magnitude, phase)

    # Assert
    np.testing.assert_array_almost_equal(reconstructed, original_stft)


# --- denormalize_spectrogram tests ---


def test_denormalize_spectrogram_minmax():
    """Test that minmax denormalization reverses normalization."""
    # Arrange
    original = np.array([[1.0, 5.0], [2.0, 8.0]])
    normalized, params = normalize_spectrogram(original, method="minmax")

    # Act
    recovered = denormalize_spectrogram(normalized, params)

    # Assert
    np.testing.assert_array_almost_equal(recovered, original)


def test_denormalize_spectrogram_none():
    """Test that 'none' method returns copy unchanged."""
    # Arrange
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    params = {"method": "none"}

    # Act
    result = denormalize_spectrogram(original, params)

    # Assert
    np.testing.assert_array_equal(result, original)


# --- compute_istft tests ---


def test_istft_output_shape():
    """Test that ISTFT produces correct output shape."""
    # Arrange - create stereo waveform, compute STFT
    rng = np.random.default_rng(5)
    original = rng.standard_normal((2, 44100), dtype=np.float32)
    stft = compute_stft(original, n_fft=4096, hop_length=1024)

    # Act
    reconstructed = compute_istft(stft, hop_length=1024, win_length=4096, length=44100)

    # Assert
    assert reconstructed.shape == (2, 44100)


def test_istft_roundtrip():
    """Test that STFT -> ISTFT approximately recovers original."""
    # Arrange - create stereo waveform
    rng = np.random.default_rng(0)
    original = rng.standard_normal((2, 44100), dtype=np.float32)

    # Act - forward and inverse
    stft = compute_stft(original, n_fft=4096, hop_length=1024)
    reconstructed = compute_istft(stft, hop_length=1024, win_length=4096, length=44100)

    # Assert - should be very close (not exact due to windowing and center=False)
    mse = np.mean((original - reconstructed) ** 2)
    # Small library/version numerical differences can move this by ~1e-4.
    assert mse < 2.5e-3, f"MSE too high: {mse}"

def test_istft_sine_wave_roundtrip():
    """Test STFT/ISTFT roundtrip with a known sine wave."""
    # Arrange - 440 Hz sine wave, stereo
    sr = 44100
    t = np.linspace(0, 1, sr, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([sine, sine])  # Same in both channels

    # Act
    stft = compute_stft(stereo, n_fft=4096, hop_length=1024)
    reconstructed = compute_istft(stft, hop_length=1024, win_length=4096, length=sr)

    # Assert
    mse = np.mean((stereo - reconstructed) ** 2)
    assert mse < 1e-3, f"MSE too high: {mse}"
