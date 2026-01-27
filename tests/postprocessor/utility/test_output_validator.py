import numpy as np
import pytest
import soundfile as sf

from src.postprocessing.utility import constants as const
from src.postprocessing.utility.output_validator import (
    OutputValidator,
)


@pytest.fixture
def valid_waveform():
    """A valid stereo waveform."""
    return np.random.rand(2, 44100).astype(np.float32) * 0.5


@pytest.fixture
def silent_waveform():
    """A completely silent waveform."""
    return np.zeros((2, 44100), dtype=np.float32)


@pytest.fixture
def nan_waveform():
    """A waveform containing NaN values."""
    waveform = np.random.rand(2, 44100).astype(np.float32)
    waveform[0, 1000] = np.nan
    return waveform


@pytest.fixture
def inf_waveform():
    """A waveform containing Inf values."""
    waveform = np.random.rand(2, 44100).astype(np.float32)
    waveform[0, 1000] = np.inf
    return waveform


def test_valid_waveform_passes(valid_waveform):
    """Test that a valid waveform passes validation."""
    validator = OutputValidator(valid_waveform)
    is_valid, message = validator.validate()

    assert is_valid
    assert "successful" in message.lower()


def test_nan_waveform_fails(nan_waveform):
    """Test that NaN values cause validation failure."""
    validator = OutputValidator(nan_waveform)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "NaN" in message


def test_inf_waveform_fails(inf_waveform):
    """Test that Inf values cause validation failure."""
    validator = OutputValidator(inf_waveform)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "infinite" in message.lower()


def test_mono_waveform_fails():
    """Test that mono waveform fails validation."""
    mono = np.random.rand(1, 44100).astype(np.float32)
    validator = OutputValidator(mono)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "Channel" in message or "shape" in message.lower()


def test_wrong_dimensions_fails():
    """Test that 3D array fails validation."""
    bad_shape = np.random.rand(1, 2, 44100).astype(np.float32)
    validator = OutputValidator(bad_shape)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "shape" in message.lower()


def test_too_short_fails():
    """Test that waveform shorter than MIN_SAMPLES fails."""
    short = np.random.rand(2, 100).astype(np.float32)  # Less than 441
    validator = OutputValidator(short)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "short" in message.lower()


def test_minimum_length_passes():
    """Test that waveform at exactly MIN_SAMPLES passes."""
    minimum = np.random.rand(2, const.MIN_SAMPLES).astype(np.float32) * 0.5
    validator = OutputValidator(minimum)
    is_valid, _ = validator.validate()

    assert is_valid


def test_silent_waveform_fails(silent_waveform):
    """Test that completely silent waveform fails."""
    validator = OutputValidator(silent_waveform)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "silent" in message.lower()


def test_near_silent_fails():
    """Test that near-silent waveform fails."""
    near_silent = np.ones((2, 44100), dtype=np.float32) * 1e-9
    validator = OutputValidator(near_silent)
    is_valid, message = validator.validate()

    assert not is_valid
    assert "silent" in message.lower()


def test_quiet_but_valid_passes():
    """Test that quiet but above-threshold waveform passes."""
    quiet = np.random.rand(2, 44100).astype(np.float32) * 0.001
    validator = OutputValidator(quiet)
    is_valid, _ = validator.validate()

    assert is_valid


def test_is_valid_method(valid_waveform):
    """Test is_valid() convenience method."""
    validator = OutputValidator(valid_waveform)
    assert validator.is_valid()


def test_get_validation_message(valid_waveform):
    """Test get_validation_message() convenience method."""
    validator = OutputValidator(valid_waveform)
    message = validator.get_validation_message()
    assert "successful" in message.lower()


def test_cached_result(valid_waveform):
    """Test that validation result is cached."""
    validator = OutputValidator(valid_waveform)

    result1 = validator.validate()
    result2 = validator.validate()

    assert result1 == result2


def test_validate_file_nonexistent(tmp_path):
    """Test that nonexistent file fails validation."""
    fake_path = tmp_path / "nonexistent.wav"
    is_valid, message = OutputValidator.validate_file(fake_path)

    assert not is_valid
    assert "not created" in message.lower() or "not" in message.lower()


def test_validate_file_success(tmp_path, valid_waveform):
    """Test that valid exported file passes validation."""
    file_path = tmp_path / "test.wav"
    sf.write(file_path, valid_waveform.T, 44100)

    is_valid, message = OutputValidator.validate_file(file_path)

    assert is_valid
    assert "successful" in message.lower()


def test_validate_file_wrong_sample_rate(tmp_path, valid_waveform):
    """Test that file with wrong sample rate fails."""
    file_path = tmp_path / "wrong_sr.wav"
    sf.write(file_path, valid_waveform.T, 22050)  # Wrong sample rate

    is_valid, message = OutputValidator.validate_file(file_path, expected_sr=44100)

    assert not is_valid
    assert "sample rate" in message.lower()


def test_validate_file_wrong_channels(tmp_path):
    """Test that file with wrong channel count fails."""
    mono = np.random.rand(44100).astype(np.float32)
    file_path = tmp_path / "mono.wav"
    sf.write(file_path, mono, 44100)

    is_valid, message = OutputValidator.validate_file(file_path)

    assert not is_valid
    assert "channel" in message.lower()
