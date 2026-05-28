from unittest.mock import patch

import pytest

from stemmy import constants as c
from stemmy.preprocessing.utility.audio_file_validator import AudioFileValidator


@pytest.fixture
def mock_metadata():
    return {
        c.METADATA_DURATION: 10.0,
        c.METADATA_SAMPLE_RATE: 44100,
        c.METADATA_CHANNELS: 2,
        c.METADATA_FILE_SIZE: 1.0,
        c.METADATA_FORMAT: "wav",
    }


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("stemmy.preprocessing.utility.audio_file_validator.AudioMetadataExtractor")
def test_validator_happy_path(mock_extractor, mock_access, mock_isfile, mock_metadata):
    mock_extractor_inst = mock_extractor.return_value
    mock_extractor_inst.get_all_metadata.return_value = mock_metadata

    validator = AudioFileValidator("test.wav")
    success, message = validator.validate()

    assert success is True
    assert message == c.SUCCESS_VALIDATION
    assert validator.is_valid() is True
    assert validator.get_validation_message() == c.SUCCESS_VALIDATION

    # Verify metadata retrieval
    retrieved_metadata = validator.get_metadata()
    assert retrieved_metadata == mock_metadata


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("stemmy.preprocessing.utility.audio_file_validator.AudioMetadataExtractor")
def test_validator_caching(mock_extractor, mock_access, mock_isfile, mock_metadata):
    mock_extractor_inst = mock_extractor.return_value
    mock_extractor_inst.get_all_metadata.return_value = mock_metadata

    validator = AudioFileValidator("test.wav")

    # First validation
    validator.validate()
    assert mock_extractor_inst.get_all_metadata.call_count == 1

    # Second validation should use cache
    validator.validate()
    assert mock_extractor_inst.get_all_metadata.call_count == 1

    # Reset validation should clear cache
    validator.reset_validation()
    validator.validate()
    assert mock_extractor_inst.get_all_metadata.call_count == 2


@patch("os.path.isfile", return_value=False)
def test_validator_file_not_found(mock_isfile):
    validator = AudioFileValidator("missing.wav")
    success, message = validator.validate()
    assert success is False
    assert c.ERROR_FILE_NOT_EXIST in message


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=False)
def test_validator_file_not_readable(mock_access, mock_isfile):
    validator = AudioFileValidator("unreadable.wav")
    success, message = validator.validate()
    assert success is False
    assert c.ERROR_FILE_NOT_READABLE in message


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("stemmy.preprocessing.utility.audio_file_validator.AudioMetadataExtractor")
def test_validator_metadata_error(mock_extractor, mock_access, mock_isfile):
    mock_extractor_inst = mock_extractor.return_value
    mock_extractor_inst.get_all_metadata.side_effect = Exception("Read error")

    validator = AudioFileValidator("test.wav")
    success, message = validator.validate()
    assert success is False
    assert "Read error" in message


@pytest.mark.parametrize(
    "key, value, expected_error",
    [
        (
            c.METADATA_FILE_SIZE,
            c.MAX_FILE_SIZE_MB + 1,
            c.ERROR_FILE_TOO_LARGE.format(max_size=c.MAX_FILE_SIZE_MB),
        ),
        (
            c.METADATA_FORMAT,
            "mp3",
            c.ERROR_UNSUPPORTED_FORMAT.format(
                format="mp3", supported=", ".join(c.SUPPORTED_FORMATS)
            ),
        ),
        (
            c.METADATA_DURATION,
            c.MIN_DURATION_SECONDS - 0.1,
            c.ERROR_AUDIO_TOO_SHORT.format(
                duration=c.MIN_DURATION_SECONDS - 0.1, min_duration=c.MIN_DURATION_SECONDS
            ),
        ),
        (
            c.METADATA_DURATION,
            c.MAX_DURATION_SECONDS + 1,
            c.ERROR_AUDIO_TOO_LONG.format(
                duration=c.MAX_DURATION_SECONDS + 1, max_duration=c.MAX_DURATION_SECONDS
            ),
        ),
        (
            c.METADATA_SAMPLE_RATE,
            c.MIN_SAMPLE_RATE - 1,
            c.ERROR_INVALID_SAMPLE_RATE.format(
                sample_rate=c.MIN_SAMPLE_RATE - 1,
                min_rate=c.MIN_SAMPLE_RATE,
                max_rate=c.MAX_SAMPLE_RATE,
            ),
        ),
        (
            c.METADATA_CHANNELS,
            c.MAX_CHANNELS + 1,
            c.ERROR_INVALID_CHANNELS.format(
                channels=c.MAX_CHANNELS + 1,
                min_channels=c.MIN_CHANNELS,
                max_channels=c.MAX_CHANNELS,
            ),
        ),
    ],
)
@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("stemmy.preprocessing.utility.audio_file_validator.AudioMetadataExtractor")
def test_validator_invalid_metadata(
    mock_extractor, mock_access, mock_isfile, mock_metadata, key, value, expected_error
):
    mock_extractor_inst = mock_extractor.return_value
    invalid_metadata = mock_metadata.copy()
    invalid_metadata[key] = value
    mock_extractor_inst.get_all_metadata.return_value = invalid_metadata

    validator = AudioFileValidator("test.wav")
    success, message = validator.validate()
    assert success is False
    assert expected_error in message


@patch("os.path.isfile", return_value=True)
@patch("os.access", return_value=True)
@patch("stemmy.preprocessing.utility.audio_file_validator.AudioMetadataExtractor")
def test_validator_unexpected_exception(mock_extractor, mock_access, mock_isfile, mock_metadata):
    mock_extractor_inst = mock_extractor.return_value
    mock_extractor_inst.get_all_metadata.return_value = mock_metadata

    validator = AudioFileValidator("test.wav")
    # Mocking internal method to raise unexpected exception
    with patch.object(validator, "_validate_file_size", side_effect=RuntimeError("System crash")):
        success, message = validator.validate()
        assert success is False
        assert "System crash" in message
        assert "Unexpected error" in message


@patch("stemmy.preprocessing.utility.audio_file_validator.AudioMetadataExtractor")
def test_validator_get_metadata_lazy_load(mock_extractor, mock_metadata):
    mock_extractor_inst = mock_extractor.return_value
    mock_extractor_inst.get_all_metadata.return_value = mock_metadata

    validator = AudioFileValidator("test.wav")
    assert validator._metadata is None

    metadata = validator.get_metadata()
    assert metadata == mock_metadata
    assert mock_extractor_inst.get_all_metadata.called
