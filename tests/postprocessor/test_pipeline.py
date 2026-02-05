from pathlib import Path

import numpy as np
import pytest
import torch

from src.postprocessing.pipeline import Postprocessor, SeparationResult
from src.preprocessing.pipeline import Preprocessor


@pytest.fixture
def test_audio_path():
    """Path to a test audio file."""
    return Path(__file__).parent.parent / "preprocessor" / "samples" / "plinky_key.wav"


@pytest.fixture
def preprocessed_data(test_audio_path):
    """Preprocess a test file and return tensor + metadata."""
    prep = Preprocessor()
    tensor, metadata = prep.process(test_audio_path)
    return tensor, metadata


def test_postprocessor_output_type(preprocessed_data, tmp_path):
    """Test that process() returns SeparationResult."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    # Create mock model output (identity mask = all ones)
    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T)

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert
    assert isinstance(result, SeparationResult)


def test_postprocessor_creates_files(preprocessed_data, tmp_path):
    """Test that process() creates vocal and instrumental files."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert
    assert result.vocals_path is not None
    assert result.instrumentals_path is not None
    assert result.vocals_path.exists()
    assert result.instrumentals_path.exists()


def test_postprocessor_file_naming(preprocessed_data, tmp_path):
    """Test that output files are named correctly."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "mysong")

    # Assert
    assert result.vocals_path.name == "mysong_vocals.wav"
    assert result.instrumentals_path.name == "mysong_instrumentals.wav"


def test_postprocessor_waveform_shape(preprocessed_data, tmp_path):
    """Test that output waveforms have correct shape."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert - should be stereo [2, N]
    assert result.vocals_waveform.ndim == 2
    assert result.vocals_waveform.shape[0] == 2
    assert result.instrumentals_waveform.ndim == 2
    assert result.instrumentals_waveform.shape[0] == 2


def test_postprocessor_waveform_length(preprocessed_data, tmp_path):
    """Test that output waveforms match original length."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert - should match processed_length from metadata
    assert result.vocals_waveform.shape[1] == metadata.processed_length
    assert result.instrumentals_waveform.shape[1] == metadata.processed_length


def test_postprocessor_sample_rate(preprocessed_data, tmp_path):
    """Test that result has correct sample rate."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test")

    # Assert
    assert result.sample_rate == metadata.processed_sr


def test_postprocessor_no_export(preprocessed_data, tmp_path):
    """Test that export_files=False skips file creation."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    F, T = tensor.shape[2], tensor.shape[3]
    mock_output = torch.ones(1, 2, F, T) * 0.5

    # Act
    result = post.process(mock_output, metadata, tensor, tmp_path, "test", export_files=False)

    # Assert
    assert result.vocals_path is None
    assert result.instrumentals_path is None
    assert result.vocals_waveform is not None  # Waveforms still computed
    assert result.instrumentals_waveform is not None


def test_postprocessor_identity_mask_roundtrip(preprocessed_data, tmp_path):
    """Test that identity mask (all ones) approximately reconstructs original."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    # Identity mask = all ones (keep everything)
    F, T = tensor.shape[2], tensor.shape[3]
    identity_mask = torch.ones(1, 2, F, T)

    # Act
    result = post.process(identity_mask, metadata, tensor, tmp_path, "test")

    # Assert - waveform should not be silent (should have energy)
    vocals_energy = np.mean(result.vocals_waveform**2)
    assert vocals_energy > 1e-6, "Reconstructed audio is silent"


def test_postprocessor_zero_mask_produces_silence(preprocessed_data, tmp_path):
    """Test that zero mask produces near-silent output."""
    # Arrange
    tensor, metadata = preprocessed_data
    post = Postprocessor()

    # Zero mask = silence everything
    F, T = tensor.shape[2], tensor.shape[3]
    zero_mask = torch.zeros(1, 2, F, T)

    # Act
    result = post.process(zero_mask, metadata, tensor, tmp_path, "test")

    # Assert - waveform should be near-silent
    vocals_energy = np.mean(result.vocals_waveform**2)
    assert vocals_energy < 1e-6, "Zero mask should produce silence"
