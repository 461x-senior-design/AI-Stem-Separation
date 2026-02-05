import librosa
import numpy as np

from .constants import HOP_LENGTH, N_FFT, WIN_LENGTH


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    Args:
        waveform: Input audio, shape [N] mono or [2, N] stereo
        n_fft: FFT window size (default: 4096)
        hop_length: Samples between frames (default: 1024)
        win_length: Window length (default: n_fft)

    Returns:
        Complex STFT array, shape [F, T] for mono or [2, F, T] for stereo where
        F = n_fft // 2 + 1 = 2049
    """
    return librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def split_magnitude_phase(stft_complex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split complex STFT into magnitude and phase components.

    Args:
        stft_complex: Complex STFT array, shape [F, T] or [2, F, T]

    Returns:
        Tuple of (magnitude, phase)
        - magnitude: Non-negative real values (same shape as input)
        - phase: Values in range [-pi, pi] (same shape as input)
    """
    magnitude = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    return magnitude, phase


def normalize_spectrogram(magnitude: np.ndarray, method: str = "minmax") -> tuple[np.ndarray, dict]:
    """
    Normalize spectrogram magnitude values.

    Args:
        magnitude: magnitude spectrogram, shape [F, T] or [2, F, T]
        method: "minmax" (scale to [0, 1]) or "none"

    Returns:
        Tuple of (normalized_magnitude, params)
        params contains everything needed to reverse normalization
    """
    normalized = magnitude.copy()

    if method == "none":
        return normalized, {"method": "none"}

    # This is the default method of normalization, as outlined in our design doc.
    if method == "minmax":
        min_val = magnitude.min()
        max_val = magnitude.max()

        if max_val - min_val == 0:
            return normalized, {"method": "minmax", "min": min_val, "max": max_val}

        normalized = (magnitude - min_val) / (max_val - min_val)
        return normalized, {"method": "minmax", "min": min_val, "max": max_val}

    raise ValueError(f"Unkown normalization method: {method}")
