import numpy as np
import pytest

from src.preprocessing.audio import normalize_waveform


def test_normalize_none_returns_copy():
    """Test that method='none' returns an unmodified copy."""
    # Arrange
    audio = np.array([[0.5, -0.25], [0.3, -0.1]])

    # Act
    result, params = normalize_waveform(audio, method="none")

    # Assert
    np.testing.assert_array_equal(result, audio)
    assert params == {"method": "none", "scale_factor": 1.0}


def test_normalize_none_does_not_modify_original():
    """Test that the original array is not modified."""
    # Arrange
    audio = np.array([[0.5, -0.25], [0.3, -0.1]])
    original_copy = audio.copy()

    # Act
    result, _ = normalize_waveform(audio, method="none")
    result[0, 0] = 999.0  # Modify the result

    # Assert - original should be unchanged
    np.testing.assert_array_equal(audio, original_copy)


def test_normalize_peak_scales_to_one():
    """Test that peak normalization scales max absolute value to 1.0."""
    # Arrange
    audio = np.array([[0.5, -0.25], [0.3, -0.4]])

    # Act
    result, params = normalize_waveform(audio, method="peak")

    # Assert
    assert np.abs(result).max() == pytest.approx(1.0)
    assert params["method"] == "peak"
    assert params["scale_factor"] == pytest.approx(0.5)


def test_normalize_peak_silent_audio():
    """Test that silent audio doesn't cause division by zero."""
    # Arrange
    audio = np.zeros((2, 100))

    # Act
    result, params = normalize_waveform(audio, method="peak")

    # Assert
    assert not np.any(np.isnan(result))  # No NaN values
    assert not np.any(np.isinf(result))  # No Inf values
    assert params["scale_factor"] == 1.0


def test_normalize_rms():
    """Test that RMS normalization produces unit RMS."""
    # Arrange
    audio = np.array([[0.4, -0.4, 0.4, -0.4], [0.2, -0.2, 0.2, -0.2]])

    # Act
    result, params = normalize_waveform(audio, method="rms")

    # Assert
    result_rms = np.sqrt(np.mean(result**2))
    assert result_rms == pytest.approx(1.0)
    assert params["method"] == "rms"


def test_normalize_rms_silent_audio():
    """Test that silent audio doesn't cause division by zero with RMS."""
    # Arrange
    audio = np.zeros((2, 100))

    # Act
    result, params = normalize_waveform(audio, method="rms")

    # Assert
    assert not np.any(np.isnan(result))
    assert params["scale_factor"] == 1.0


def test_normalize_invalid_method():
    """Test that invalid method raises ValueError."""
    # Arrange
    audio = np.array([[0.5, -0.25], [0.3, -0.1]])

    # Act & Assert
    with pytest.raises(ValueError):
        normalize_waveform(audio, method="invalid")
