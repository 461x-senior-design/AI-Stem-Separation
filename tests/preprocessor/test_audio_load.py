import pytest
import numpy as np
from unittest.mock import patch
from stemmy.preprocessing.audio import load_audio
from stemmy.constants import TARGET_SAMPLE_RATE

def test_load_audio_file_not_found():
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio("non_existent_file.wav")

def test_load_audio_not_a_file(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="Audio path is not a file"):
        load_audio(d)

def test_load_audio_invalid_sr(tmp_path):
    f = tmp_path / "test.wav"
    f.touch()
    with pytest.raises(ValueError, match="sr must be a positive integer"):
        load_audio(f, sr=0)
    with pytest.raises(ValueError, match="sr must be a positive integer"):
        load_audio(f, sr=-1)
    with pytest.raises(ValueError, match="sr must be a positive integer"):
        load_audio(f, sr="44100")

def test_load_audio_invalid_mono(tmp_path):
    f = tmp_path / "test.wav"
    f.touch()
    with pytest.raises(TypeError, match="mono must be a bool"):
        load_audio(f, mono="True")

@patch("librosa.load")
def test_load_audio_happy_path(mock_load, tmp_path):
    f = tmp_path / "test.wav"
    f.touch()
    mock_load.return_value = (np.zeros((2, 100)), 44100)
    
    waveform, sr = load_audio(f, sr=22050, mono=True)
    
    assert sr == 22050
    mock_load.assert_called_once_with(str(f), sr=22050, mono=True)

def test_load_audio_path_obj(tmp_path):
    f = tmp_path / "test.wav"
    f.touch()
    with patch("librosa.load", return_value=(np.zeros(100), 44100)):
        load_audio(f)
