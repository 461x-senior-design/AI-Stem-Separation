import numpy as np
import pytest

from stemmy.preprocessing.spectral import compute_stft, normalize_spectrogram, split_magnitude_phase


def test_stft_output_shape_mono():
    """
    Test that STFT produces correct output shape for mono input.
    """
    # Arrange - 1 second of audio at 44100 Hz
    waveform = np.random.randn(44100).astype(np.float32)

    # Act
    stft = compute_stft(waveform)

    # Assert
    assert stft.shape[0] == 2049  # F = n_fft // 2 + 1
    assert stft.ndim == 2


def test_split_magnitude_phase_values():
    """Test that magnitude and phase are computed
    correctly."""
    # Arrange - known complex values
    stft = np.array([1 + 1j, 2 + 0j, 0 + 3j])

    # Act
    mag, phase = split_magnitude_phase(stft)

    # Assert
    assert mag[0] == pytest.approx(np.sqrt(2))  # |1+1j| = sqrt(2)
    assert mag[1] == pytest.approx(2.0)  # |2+0j| = 2
    assert mag[2] == pytest.approx(3.0)  # |0+3j| = 3
    assert phase[1] == pytest.approx(0.0)  # angle(2+0j) = 0


def test_magnitude_is_non_negative():
    """Test that magnitude values are never negative."""
    # Arrange
    waveform = np.random.randn(44100).astype(np.float32)
    stft = compute_stft(waveform)

    # Act
    mag, _ = split_magnitude_phase(stft)

    # Assert
    assert np.all(mag >= 0)


def test_normalize_spectrogram_none():
    """Test that method='none' returns unchanged copy."""
    # Arrange
    mag = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Act
    result, params = normalize_spectrogram(mag, method="none")

    # Assert
    np.testing.assert_array_equal(result, mag)
    assert params["method"] == "none"


def test_normalize_spectrogram_minmax():
    """Test that minmax scales to [0, 1] range."""
    # Arrange
    mag = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Act
    result, params = normalize_spectrogram(mag, method="minmax")

    # Assert
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)
    assert params["min"] == pytest.approx(1.0)
    assert params["max"] == pytest.approx(4.0)


def test_normalize_spectrogram_minmax_constant():
    """Test that constant spectrogram doesn't cause division by zero."""
    # Arrange
    mag = np.ones((10, 10)) * 5.0

    # Act
    result, params = normalize_spectrogram(mag, method="minmax")

    # Assert
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_normalize_spectrogram_invalid_method():
    """Test that invalid method raises ValueError."""
    # Arrange
    mag = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Act & Assert
    with pytest.raises(ValueError):
        normalize_spectrogram(mag, method="invalid")
