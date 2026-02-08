from pathlib import Path

import pytest
import torch

from stemmy.preprocessing.pipeline import PreprocessingMetadata, Preprocessor


@pytest.fixture
def test_audio_path():
    """Path to a test audio file."""
    return Path(__file__).parent / "samples" / "plinky_key.wav"


def test_preprocessor_output_types(test_audio_path):
    """Test that process() returns correct types."""
    # Arrange
    prep = Preprocessor()

    # Act
    tensor, metadata = prep.process(test_audio_path)

    # Assert
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(metadata, PreprocessingMetadata)


def test_preprocessor_tensor_shape(test_audio_path):
    """Test that output tensor has correct shape [1, 2, F, T]."""
    # Arrange
    prep = Preprocessor()

    # Act
    tensor, metadata = prep.process(test_audio_path)

    # Assert
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1  # batch
    assert tensor.shape[1] == 2  # stereo channels
    assert tensor.shape[2] == 2049  # frequency bins (n_fft // 2 + 1)
    assert tensor.shape[3] > 0  # time frames


def test_preprocessor_tensor_dtype(test_audio_path):
    """Test that tensor is float32."""
    # Arrange
    prep = Preprocessor()

    # Act
    tensor, _ = prep.process(test_audio_path)

    # Assert
    assert tensor.dtype == torch.float32


def test_preprocessor_metadata_has_phase(test_audio_path):
    """Test that metadata contains phase array."""
    # Arrange
    prep = Preprocessor()

    # Act
    tensor, metadata = prep.process(test_audio_path)

    # Assert
    assert metadata.phase is not None
    assert metadata.phase.shape[0] == 2  # stereo
    assert metadata.phase.shape[1] == 2049  # frequency bins
    assert metadata.phase.shape[2] == tensor.shape[3]  # same time frames


def test_preprocessor_metadata_has_norm_params(test_audio_path):
    """Test that metadata contains normalization parameters."""
    # Arrange
    prep = Preprocessor()

    # Act
    _, metadata = prep.process(test_audio_path)

    # Assert
    assert "scale_factor" in metadata.waveform_norm_params
    assert "min" in metadata.spectrogram_norm_params
    assert "max" in metadata.spectrogram_norm_params


def test_preprocessor_normalized_range(test_audio_path):
    """Test that spectrogram values are in [0, 1] after minmax normalization."""
    # Arrange
    prep = Preprocessor(spectrogram_norm="minmax")

    # Act
    tensor, _ = prep.process(test_audio_path)

    # Assert
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
