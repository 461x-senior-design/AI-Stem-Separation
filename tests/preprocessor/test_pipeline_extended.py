from unittest.mock import patch

import numpy as np
import pytest

from stemmy.preprocessing.pipeline import Preprocessor
from stemmy.preprocessing.utility.audio_file_validator import AudioValidationException


def test_preprocessor_init_invalid():
    with pytest.raises(ValueError, match="target_channels must be 2"):
        Preprocessor(target_channels=1)
    with pytest.raises(ValueError, match="audio_channels must be 2"):
        Preprocessor(audio_channels=1)

@patch("stemmy.preprocessing.pipeline.AudioFileValidator")
def test_preprocessor_invalid_file(mock_validator):
    mock_val_inst = mock_validator.return_value
    mock_val_inst.validate.return_value = (False, "Invalid format")
    
    prep = Preprocessor()
    with pytest.raises(AudioValidationException, match="Invalid format"):
        prep.process("dummy.txt")

@patch("stemmy.preprocessing.pipeline.AudioFileValidator")
@patch("stemmy.preprocessing.pipeline.load_audio")
@patch("stemmy.preprocessing.pipeline.ensure_stereo")
def test_preprocessor_ensure_stereo_failure(mock_stereo, mock_load, mock_validator):
    mock_validator.return_value.validate.return_value = (True, "Valid")
    mock_load.return_value = (np.zeros((1, 100)), 44100)
    mock_stereo.return_value = np.zeros((3, 100)) # Invalid channels
    
    prep = Preprocessor()
    with pytest.raises(ValueError, match="Expected stereo waveform"):
        prep.process("dummy.wav")

@patch("stemmy.preprocessing.pipeline.AudioFileValidator")
@patch("stemmy.preprocessing.pipeline.load_audio")
@patch("stemmy.preprocessing.pipeline.normalize_waveform")
def test_preprocessor_zero_scale(mock_norm, mock_load, mock_validator):
    mock_validator.return_value.validate.return_value = (True, "Valid")
    mock_load.return_value = (np.zeros((2, 10000)), 44100)
    mock_norm.return_value = (np.zeros(10000), {"scale_factor": 0.0})
    
    prep = Preprocessor()
    # Should not crash, scale becomes 1.0
    tensor, _ = prep.process("dummy.wav")
    assert tensor.shape[1] == 2

@patch("stemmy.preprocessing.pipeline.AudioFileValidator")
@patch("stemmy.preprocessing.pipeline.load_audio")
@patch("stemmy.preprocessing.pipeline.compute_stft")
@patch("stemmy.preprocessing.pipeline.split_magnitude_phase")
def test_preprocessor_magnitude_shape_failure(mock_split, mock_stft, mock_load, mock_validator):
    mock_validator.return_value.validate.return_value = (True, "Valid")
    mock_load.return_value = (np.zeros((2, 10000)), 44100)
    mock_split.return_value = (np.zeros((1, 10, 10)), np.zeros((1, 10, 10))) # Only 1 channel
    
    prep = Preprocessor()
    with pytest.raises(ValueError, match="Unexpected magnitude shape"):
        prep.process("dummy.wav")

@patch("stemmy.preprocessing.pipeline.AudioFileValidator")
@patch("stemmy.preprocessing.pipeline.load_audio")
@patch("stemmy.preprocessing.pipeline.normalize_spectrogram")
def test_preprocessor_norm_magnitude_shape_failure(mock_norm, mock_load, mock_validator):
    mock_validator.return_value.validate.return_value = (True, "Valid")
    mock_load.return_value = (np.zeros((2, 10000)), 44100)
    mock_norm.return_value = (np.zeros((4, 10, 10)), {}) # 4 channels
    
    prep = Preprocessor()
    with pytest.raises(ValueError, match="Unexpected normalized magnitude shape"):
        prep.process("dummy.wav")
