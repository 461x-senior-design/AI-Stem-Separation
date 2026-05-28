import pytest
from unittest.mock import patch, MagicMock
from stemmy.preprocessing.utility.audio_metadata_extractor import AudioMetadataExtractor
from stemmy import constants as c

@patch("librosa.get_duration")
@patch("soundfile.info")
@patch("os.path.getsize")
def test_extractor_get_all_metadata(mock_getsize, mock_sf_info, mock_duration):
    mock_duration.return_value = 10.0
    mock_sf_info.return_value = MagicMock(samplerate=44100, channels=2)
    mock_getsize.return_value = 1024 * 1024 # 1MB
    
    extractor = AudioMetadataExtractor("test.wav")
    metadata = extractor.get_all_metadata()
    
    assert metadata[c.METADATA_DURATION] == 10.0
    assert metadata[c.METADATA_SAMPLE_RATE] == 44100
    assert metadata[c.METADATA_CHANNELS] == 2
    assert metadata[c.METADATA_FILE_SIZE] == 1.0
    assert metadata[c.METADATA_FORMAT] == "wav"

@patch("librosa.get_duration")
@patch("soundfile.info")
@patch("os.path.getsize")
def test_extractor_caching(mock_getsize, mock_sf_info, mock_duration):
    mock_duration.return_value = 10.0
    mock_sf_info.return_value = MagicMock(samplerate=44100, channels=2)
    mock_getsize.return_value = 1024 * 1024
    
    extractor = AudioMetadataExtractor("test.wav")
    
    # First call - should call mocks
    extractor.get_all_metadata()
    assert mock_duration.call_count == 1
    
    # Second call - should use cache
    extractor.get_all_metadata()
    assert mock_duration.call_count == 1
    
    # Clear cache
    extractor.clear_cache()
    
    # Third call - should call mocks again
    extractor.get_all_metadata()
    assert mock_duration.call_count == 2

def test_extractor_get_format():
    extractor = AudioMetadataExtractor("path/to/test.FLAC")
    assert extractor.get_format() == "flac"
    
    extractor = AudioMetadataExtractor("no_extension")
    assert extractor.get_format() == ""
