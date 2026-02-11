import librosa
import numpy as np

#########################
# Change by Ryan:
# Reason:
# Shared STFT defaults live in the project-wide constants module so preprocessing
# stays aligned with training/inference framing and defaults.
# What it does:
# Imports STFT defaults (N_FFT/HOP_LENGTH/WIN_LENGTH/STFT_CENTER) from stemmy.constants.
from stemmy.constants import HOP_LENGTH, N_FFT, STFT_CENTER, WIN_LENGTH

#########################


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    #########################
    # Change by Ryan:
    # Reason:
    # Include center to match the framing used by training/inference.
    # What it does:
    # Adds a center parameter with default from stemmy.constants.STFT_CENTER.
    center: bool = STFT_CENTER,
    #########################
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    Args:
        waveform: Input audio, shape [N] mono or [2, N] stereo
        n_fft: FFT window size (default: 4096)
        hop_length: Samples between frames (default: 1024)
        win_length: Window length (default: n_fft)
        #########################
        # Change by Ryan:
        # Reason:
        # Document the center parameter because it is part of the framing contract.
        # What it does:
        # Adds center to the Args list so docstring matches the function signature.
        center: Whether to pad the signal so frames are centered (must match project setting)
        #########################

    Returns:
        Complex STFT array, shape [F, T] for mono or [2, F, T] for stereo where
        F = n_fft // 2 + 1 = 2049
    """
    #########################
    # Change by Ryan:
    # Reason:
    # librosa.stft expects a 1-D time series. Passing a 2-D stereo array is incorrect.
    # This function's docstring says it supports stereo [2, N], so we must explicitly
    # handle stereo by computing per-channel STFT and stacking.
    # What it does:
    # Supports waveform shapes:
    #   - [N] mono
    #   - [2, N] stereo (channel-first)
    #   - [N, 2] stereo (channel-last)
    # and returns [F, T] for mono or [2, F, T] for stereo.
    if not isinstance(waveform, np.ndarray):
        raise ValueError("waveform must be a numpy.ndarray.")

    if waveform.ndim == 1:
        return librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=bool(center),
        )

    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [N] or stereo shape [2, N] or [N, 2].")

    # Support both channel-first and channel-last stereo layouts.
    if waveform.shape[0] == 2:
        ch0 = waveform[0]
        ch1 = waveform[1]
    elif waveform.shape[1] == 2:
        ch0 = waveform[:, 0]
        ch1 = waveform[:, 1]
    else:
        raise ValueError("Stereo waveform must have 2 channels (shape [2, N] or [N, 2]).")

    stft0 = librosa.stft(
        ch0,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=bool(center),
    )
    stft1 = librosa.stft(
        ch1,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=bool(center),
    )
    return np.stack([stft0, stft1], axis=0)
    #########################


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
        method: "minmax" (scale to [0, 1]), "freq_minmax" (per-frequency scaling), or "none"

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

    #########################
    # Change by Ryan:
    # Reason:
    # Add per-frequency min/max normalization to match the project-wide default
    # input scaling used by training/inference ("freq_minmax").
    # The unified pipeline normalizes the mono magnitude [F, T] that is fed to the model.
    # What it does:
    # Implements a per-frequency (per-row) min/max normalization for [F, T] inputs and
    # returns the f_min/f_max parameters needed to reverse the normalization.
    if method == "freq_minmax":
        if magnitude.ndim != 2:
            raise ValueError("freq_minmax expects magnitude with shape [F, T].")

        f_min = magnitude.min(axis=1, keepdims=True)  # [F, 1]
        f_max = magnitude.max(axis=1, keepdims=True)  # [F, 1]

        eps = (
            np.finfo(magnitude.dtype).eps if np.issubdtype(magnitude.dtype, np.floating) else 1e-12
        )
        denom = np.maximum(f_max - f_min, eps)
        normalized = np.clip((magnitude - f_min) / denom, 0.0, 1.0)
        return normalized, {"method": "freq_minmax", "f_min": f_min, "f_max": f_max}

    raise ValueError(f"Unknown normalization method: {method}")
    #########################
