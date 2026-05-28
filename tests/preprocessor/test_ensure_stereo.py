import numpy as np
import pytest

from stemmy.preprocessing.audio import ensure_stereo


def test_ensure_stereo_from_mono():
    """Test that a 1D mono array is converted to a 2D stereo array."""
    # Arrange
    mono = np.array([0.1, 0.2, 0.3, 0.4])

    # Act
    stereo = ensure_stereo(mono)

    # Assert
    assert stereo.ndim == 2
    assert stereo.shape == (2, 4)
    # Ensure the audio content was duplicated correctly across both channels
    np.testing.assert_array_equal(stereo[0], mono)
    np.testing.assert_array_equal(stereo[1], mono)


def test_ensure_stereo_passthrough():
    """Test that an already stereo array is returned unchanged."""
    # Arrange
    already_stereo = np.array([[0.1, 0.2], [0.3, 0.4]])

    # Act
    result = ensure_stereo(already_stereo)

    # Assert
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, already_stereo)


@pytest.mark.parametrize(
    "invalid_input",
    [
        np.zeros((3, 100)),  # 3 channels (not supported)
        np.zeros((2, 2, 100)),  # 3D array (not supported)
    ],
)
def test_ensure_stereo_invalid_shapes(invalid_input):
    """Optional: Test that the function raises an error for unsupported dimensions."""
    with pytest.raises(ValueError):
        ensure_stereo(invalid_input)


def test_ensure_stereo_invalid_type():
    """Test that the function raises a TypeError for non-numpy array input."""
    with pytest.raises(TypeError, match="waveform must be a numpy.ndarray"):
        ensure_stereo([1, 2, 3])
