import numpy as np
import pytest

from stemmy.constants import FREQ_MINMAX_EPS, N_FFT, WINDOW
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
    magnitude = np.random.rand(2, 100, 50)
    mask = np.ones_like(magnitude)

    # Act
    result = apply_mask(magnitude, mask)

    # Assert
    np.testing.assert_array_equal(result, magnitude)


def test_apply_mask_zeros_silences():
    """Test that mask of all zeros returns zeros."""
    # Arrange
    magnitude = np.random.rand(2, 100, 50)
    mask = np.zeros_like(magnitude)

    # Act
    result = apply_mask(magnitude, mask)

    # Assert
    assert np.all(result == 0)


def test_combine_magnitude_phase_basic():
    """Test that magnitude and phase combine correctly."""
    # Arrange - known values
    magnitude = np.array([2.0, 1.0, 3.0])
    phase = np.array([0.0, np.pi / 2, np.pi])

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
    magnitude = np.random.rand(2, 100, 50)
    phase = np.random.uniform(-np.pi, np.pi, (2, 100, 50))

    # Act
    result = combine_magnitude_phase(magnitude, phase)

    # Assert
    assert np.iscomplexobj(result)


def test_combine_magnitude_phase_preserves_magnitude():
    """Test that |combine(mag, phase)| == mag."""
    # Arrange
    magnitude = np.random.rand(2, 100, 50)
    phase = np.random.uniform(-np.pi, np.pi, (2, 100, 50))

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


def test_denormalize_spectrogram_freq_minmax():
    """Test that freq_minmax denormalization restores per-frequency values."""
    # Arrange
    normalized = np.array(
        [
            [[0.0, 0.5, 1.0], [0.25, 0.5, 0.75]],
            [[1.0, 0.5, 0.0], [0.75, 0.5, 0.25]],
        ],
        dtype=np.float32,
    )
    f_min = np.array(
        [
            [[1.0], [10.0]],
            [[2.0], [20.0]],
        ],
        dtype=np.float32,
    )
    f_max = np.array(
        [
            [[5.0], [18.0]],
            [[6.0], [28.0]],
        ],
        dtype=np.float32,
    )
    params = {
        "method": "freq_minmax",
        "f_min": f_min,
        "f_max": f_max,
    }

    # Act
    result = denormalize_spectrogram(normalized, params)

    # Assert
    expected = normalized * (f_max - f_min) + f_min
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_denormalize_spectrogram_freq_minmax_zero_range_uses_epsilon():
    """Test that freq_minmax handles a frequency bin with zero range."""
    # Arrange
    normalized = np.ones((2, 2, 3), dtype=np.float32)
    f_min = np.full((2, 2, 1), 4.0, dtype=np.float32)
    f_max = np.full((2, 2, 1), 4.0, dtype=np.float32)
    params = {
        "method": "freq_minmax",
        "f_min": f_min,
        "f_max": f_max,
    }

    # Act
    result = denormalize_spectrogram(normalized, params)

    # Assert
    expected = normalized * FREQ_MINMAX_EPS + f_min
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_denormalize_spectrogram_unknown_method_raises_value_error():
    """Test that an unsupported normalization method raises ValueError."""
    # Arrange
    normalized = np.zeros((2, 2, 2), dtype=np.float32)

    # Act and Assert
    with pytest.raises(ValueError, match="Unknown normalization method"):
        denormalize_spectrogram(normalized, {"method": "unsupported"})


# --- compute_istft tests ---


def test_istft_output_shape():
    """Test that ISTFT produces correct output shape."""
    # Arrange - create stereo waveform, compute STFT
    original = np.random.randn(2, 44100).astype(np.float32)
    stft = compute_stft(original, n_fft=4096, hop_length=1024)

    # Act
    reconstructed = compute_istft(stft, hop_length=1024, win_length=4096, length=44100)

    # Assert
    assert reconstructed.shape == (2, 44100)


def test_istft_roundtrip():
    """Test that STFT -> ISTFT approximately recovers original."""
    # Arrange - create stereo waveform
    np.random.seed(0)
    original = np.random.randn(2, 44100).astype(np.float32)

    # Act - forward and inverse
    stft = compute_stft(original, n_fft=4096, hop_length=1024)
    reconstructed = compute_istft(stft, hop_length=1024, win_length=4096, length=44100)

    # Assert - should be very close (not exact due to windowing and center=False)
    mse = np.mean((original - reconstructed) ** 2)
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


def test_istft_rejects_non_string_window():
    """Test that ISTFT rejects a non-string window value."""
    # Arrange
    stft = np.ones((2, 1 + (N_FFT // 2), 4), dtype=np.complex64)

    # Act and Assert
    with pytest.raises(ValueError, match="window must be a non-empty string"):
        compute_istft(stft, hop_length=1024, win_length=4096, window=123)


def test_istft_rejects_empty_window():
    """Test that ISTFT rejects an empty window name."""
    # Arrange
    stft = np.ones((2, 1 + (N_FFT // 2), 4), dtype=np.complex64)

    # Act and Assert
    with pytest.raises(ValueError, match="window must be a non-empty string"):
        compute_istft(stft, hop_length=1024, win_length=4096, window="   ")


def test_istft_rejects_window_that_does_not_match_project_constant():
    """Test that ISTFT rejects a window other than the project window."""
    # Arrange
    stft = np.ones((2, 1 + (N_FFT // 2), 4), dtype=np.complex64)

    # Act and Assert
    with pytest.raises(ValueError, match="window must match stemmy.constants.WINDOW"):
        compute_istft(stft, hop_length=1024, win_length=4096, window="boxcar")


def test_istft_rejects_single_channel_stft():
    """Test that ISTFT rejects a complex STFT without two channels."""
    # Arrange
    stft = np.ones((1, 1 + (N_FFT // 2), 4), dtype=np.complex64)

    # Act and Assert
    with pytest.raises(ValueError, match="stft_complex must have shape"):
        compute_istft(stft, hop_length=1024, win_length=4096, window=WINDOW)


def test_istft_rejects_stft_without_channel_dimension():
    """Test that ISTFT rejects a two-dimensional STFT array."""
    # Arrange
    stft = np.ones((1 + (N_FFT // 2), 4), dtype=np.complex64)

    # Act and Assert
    with pytest.raises(ValueError, match="stft_complex must have shape"):
        compute_istft(stft, hop_length=1024, win_length=4096, window=WINDOW)
