# postprocessing/spectral.py
# Frequency domain operations including
# - Mask application
# - Spectogram denormalizaiton

import librosa
import numpy as np


def apply_mask(magnitude: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a soft mask to a magnitude spectrogram.

    Args:
        magnitude: Original magnitude spectrogram, shape [2, F, T]
        mask: Mask values in [0, 1], same shape as magnitude

    Returns:
        Masked magnitude spectrogram (magnitude * mask)
    """
    return magnitude * mask


def combine_magnitude_phase(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Combines the magnitude and phase into complex STFT.

    Args:
        magnitude: Magnitude spectrogram, shapes [2, F, T]
        phase: Phase in radians, shape [2, F, T]

    Returns:
        Complex STFT array, shape [2, F, T]
    """
    return magnitude * np.exp(1j * phase)


def denormalize_spectrogram(normalized: np.ndarray, params: dict) -> np.ndarray:
    """
    Reverse spectrogram normalization.

    Args:
        normalized: Normalized spectrogram, shape [2, F, T]
        params: Parameters from normalize_spectrogram()

    Returns:
        Original-scale magnitude spectrogram
    """
    method = params.get("method")

    if method == "none":
        return normalized.copy()

    if method == "minmax":
        min_val = params["min"]
        max_val = params["max"]
        return normalized * (max_val - min_val) + min_val

    raise ValueError(f"Unknown normalization method: {method}")


def compute_istft(
    stft_complex: np.ndarray, hop_length: int, win_length: int, length: int = None
) -> np.ndarray:
    """
    Compute Inverse Short-Time Fourier Transform.

    Args:
        stft_complex: Complex STFT array, shape [2, F, T]
        hop_length: Hop length (must match forward STFT)
        win_length: Window length (must match forward STFT)
        length: Output length in samples (trims/pads to match)

    Returns:
        Time-domain waveform, shape [2, N] for stereo
    """
    # Process each channel
    left = librosa.istft(
        stft_complex[0], hop_length=hop_length, win_length=win_length, length=length
    )
    right = librosa.istft(
        stft_complex[1], hop_length=hop_length, win_length=win_length, length=length
    )

    return np.stack([left, right])
