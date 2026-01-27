
import numpy as np
import soundfile as sf

from src.postprocessing.audio import denormalize_waveform, export_audio
from src.preprocessing.audio import normalize_waveform

# --- denormalize_waveform tests ---


def test_denormalize_waveform_peak():
    """Test that peak denormalization reverses normalization."""
    # Arrange
    original = np.array([[0.5, -0.25], [0.3, -0.4]])
    normalized, params = normalize_waveform(original, method="peak")

    # Act
    recovered = denormalize_waveform(normalized, params)

    # Assert
    np.testing.assert_array_almost_equal(recovered, original)


def test_denormalize_waveform_rms():
    """Test that rms denormalization reverses normalization."""
    # Arrange
    original = np.array([[0.5, -0.25, 0.3], [0.3, -0.4, 0.2]])
    normalized, params = normalize_waveform(original, method="rms")

    # Act
    recovered = denormalize_waveform(normalized, params)

    # Assert
    np.testing.assert_array_almost_equal(recovered, original)


def test_denormalize_waveform_none():
    """Test that 'none' method returns copy."""
    # Arrange
    original = np.array([[0.5, -0.25], [0.3, -0.4]])
    params = {"method": "none", "scale_factor": 1.0}

    # Act
    result = denormalize_waveform(original, params)

    # Assert
    np.testing.assert_array_equal(result, original)


# --- export_audio tests ---


def test_export_audio_creates_file(tmp_path):
    """Test that export_audio creates a file."""
    # Arrange
    waveform = np.random.rand(2, 44100).astype(np.float32) * 2 - 1  # [-1, 1]
    output_path = tmp_path / "test_output.wav"

    # Act
    result_path = export_audio(waveform, output_path, sample_rate=44100)

    # Assert
    assert result_path.exists()
    assert result_path == output_path


def test_export_audio_correct_sample_rate(tmp_path):
    """Test that exported file has correct sample rate."""
    # Arrange
    waveform = np.random.rand(2, 44100).astype(np.float32) * 2 - 1
    output_path = tmp_path / "test_output.wav"

    # Act
    export_audio(waveform, output_path, sample_rate=44100)

    # Assert
    info = sf.info(output_path)
    assert info.samplerate == 44100


def test_export_audio_correct_channels(tmp_path):
    """Test that exported file has correct channel count."""
    # Arrange
    waveform = np.random.rand(2, 44100).astype(np.float32) * 2 - 1
    output_path = tmp_path / "test_output.wav"

    # Act
    export_audio(waveform, output_path, sample_rate=44100)

    # Assert
    info = sf.info(output_path)
    assert info.channels == 2


def test_export_audio_correct_length(tmp_path):
    """Test that exported file has correct length."""
    # Arrange
    num_samples = 44100
    waveform = np.random.rand(2, num_samples).astype(np.float32) * 2 - 1
    output_path = tmp_path / "test_output.wav"

    # Act
    export_audio(waveform, output_path, sample_rate=44100)

    # Assert
    info = sf.info(output_path)
    assert info.frames == num_samples


def test_export_audio_creates_parent_dirs(tmp_path):
    """Test that export_audio creates parent directories."""
    # Arrange
    waveform = np.random.rand(2, 1000).astype(np.float32) * 2 - 1
    output_path = tmp_path / "nested" / "dirs" / "test.wav"

    # Act
    result_path = export_audio(waveform, output_path, sample_rate=44100)

    # Assert
    assert result_path.exists()


def test_export_audio_clips_values(tmp_path):
    """Test that values outside [-1, 1] are clipped."""
    # Arrange - waveform with values outside [-1, 1]
    waveform = np.array([[1.5, -1.5, 0.5], [2.0, -2.0, 0.0]])
    output_path = tmp_path / "test_clipped.wav"

    # Act
    export_audio(waveform, output_path, sample_rate=44100)

    # Assert - read back and verify clipped
    loaded, sr = sf.read(output_path)
    assert loaded.max() <= 1.0
    assert loaded.min() >= -1.0


def test_export_audio_roundtrip(tmp_path):
    """Test that exported audio can be read back correctly."""
    # Arrange
    original = np.random.rand(2, 44100).astype(np.float32) * 2 - 1
    output_path = tmp_path / "test_roundtrip.wav"

    # Act
    export_audio(original, output_path, sample_rate=44100)
    loaded, sr = sf.read(output_path)
    loaded = loaded.T  # Convert back to [channels, samples]

    # Assert - PCM_16 has some quantization error, so use tolerance
    np.testing.assert_array_almost_equal(loaded, original, decimal=4)
