import numpy as np
import pytest

from stemmy.constants import WINDOW
from stemmy.preprocessing.spectral import compute_stft, normalize_spectrogram, split_magnitude_phase


def test_compute_stft_invalid_input():
    with pytest.raises(TypeError, match="waveform must be a numpy.ndarray"):
        compute_stft([1, 2, 3])

    with pytest.raises(ValueError, match="n_fft must be a positive integer"):
        compute_stft(np.zeros(100), n_fft=0)

    with pytest.raises(ValueError, match="hop_length must be a positive integer"):
        compute_stft(np.zeros(100), hop_length=-1)

    with pytest.raises(ValueError, match="win_length must be a positive integer"):
        compute_stft(np.zeros(100), win_length="1024")

    with pytest.raises(TypeError, match="center must be a bool"):
        compute_stft(np.zeros(100), center=1)

    with pytest.raises(ValueError, match="window must be a non-empty string"):
        compute_stft(np.zeros(100), window="")

    with pytest.raises(ValueError, match="window must match stemmy.constants.WINDOW"):
        compute_stft(np.zeros(100), window="hamming" if WINDOW == "hann" else "hann")


def test_compute_stft_stereo_shapes():
    # Channel first [2, N]
    wav_cf = np.zeros((2, 44100))
    stft_cf = compute_stft(wav_cf)
    assert stft_cf.shape[0] == 2

    # Channel last [N, 2]
    wav_cl = np.zeros((44100, 2))
    stft_cl = compute_stft(wav_cl)
    assert stft_cl.shape[0] == 2

    # Invalid stereo shape [3, N]
    with pytest.raises(ValueError, match="Stereo waveform must have shape"):
        compute_stft(np.zeros((3, 44100)))

    # Invalid rank [1, 1, N]
    with pytest.raises(ValueError, match="waveform must have shape"):
        compute_stft(np.zeros((1, 1, 100)))


def test_split_magnitude_phase_errors():
    with pytest.raises(TypeError, match="stft_complex must be a numpy.ndarray"):
        split_magnitude_phase([1j, 2j])
    with pytest.raises(ValueError, match="stft_complex must have shape"):
        split_magnitude_phase(np.zeros((1, 1, 1, 1)))


def test_normalize_spectrogram_errors():
    with pytest.raises(TypeError, match="magnitude must be a numpy.ndarray"):
        normalize_spectrogram([1, 2])
    with pytest.raises(ValueError, match="magnitude must have shape"):
        normalize_spectrogram(np.zeros(100))
    with pytest.raises(ValueError, match="method must be a non-empty string"):
        normalize_spectrogram(np.zeros((10, 10)), method=None)
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_spectrogram(np.zeros((10, 10)), method="log")


def test_normalize_spectrogram_minmax_zero_div():
    mag = np.zeros((10, 10))
    norm, params = normalize_spectrogram(mag, method="minmax")
    assert np.all(norm == 0)
    assert params["min"] == 0
    assert params["max"] == 0


def test_normalize_spectrogram_none():
    mag = np.random.rand(10, 10)
    norm, params = normalize_spectrogram(mag, method="none")
    assert np.array_equal(norm, mag)
    assert params["method"] == "none"
