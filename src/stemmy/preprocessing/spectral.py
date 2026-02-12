import librosa
import numpy as np

#########################
# Change by Ryan:
# Reason:
# Shared STFT defaults live in the project-wide constants module so preprocessing
# stays aligned with training/inference framing and defaults.
# What it does:
# - Imports STFT defaults (N_FFT/HOP_LENGTH/WIN_LENGTH/STFT_CENTER/WINDOW) from stemmy.constants.
# - Standardizes freq_minmax epsilon floor to 1e-8 to match training (avoids distribution shift).
from stemmy.constants import HOP_LENGTH, N_FFT, STFT_CENTER, WIN_LENGTH, WINDOW

_FREQ_MINMAX_EPS = 1e-8

#########################


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    center: bool = STFT_CENTER,
    window: str = WINDOW,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    Args:
        waveform: Input audio, shape [N] mono, [2, N] stereo (channel-first),
                  or [N, 2] stereo (channel-last)
        n_fft: FFT window size (default: 4096)
        hop_length: Samples between frames (default: 1024)
        win_length: Window length (default: n_fft)
        center: Whether to pad so frames are centered (must match training/inference)
        window: Window function name (must match stemmy.constants.WINDOW)

    Returns:
        Complex STFT array, shape [F, T] for mono or [2, F, T] for stereo where
        F = n_fft // 2 + 1 = 2049
    """
    #########################
    # Change by Ryan:
    # Reason:
    # Tighten input validation and explicitly expose/pass `center` so STFT framing matches
    # the canonical training/inference contract.
    # What it does:
    # - Validates waveform is a numpy array.
    # - Validates n_fft/hop_length/win_length are positive ints.
    # - Validates center is a bool.
    if not isinstance(waveform, np.ndarray):
        raise TypeError("waveform must be a numpy.ndarray.")
    if not isinstance(n_fft, int) or n_fft <= 0:
        raise ValueError("n_fft must be a positive integer.")
    if not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError("hop_length must be a positive integer.")
    if not isinstance(win_length, int) or win_length <= 0:
        raise ValueError("win_length must be a positive integer.")
    if not isinstance(center, (bool, np.bool_)):
        raise TypeError("center must be a bool.")
    if not isinstance(window, str) or window.strip() == "":
        raise ValueError("window must be a non-empty string.")
    #########################

    window = window.strip().lower()
    expected_window = str(WINDOW).strip().lower()
    if window != expected_window:
        raise ValueError("window must match stemmy.constants.WINDOW (%s)." % (str(WINDOW),))

    if waveform.ndim == 1:
        return librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            window=window,
        )

    if waveform.ndim == 2:
        # Accept both channel-first [2, N] and channel-last [N, 2]
        if waveform.shape[0] == 2:
            left = librosa.stft(
                waveform[0],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=center,
                window=window,
            )
            right = librosa.stft(
                waveform[1],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=center,
                window=window,
            )
            return np.stack([left, right], axis=0)

        if waveform.shape[1] == 2:
            left = librosa.stft(
                waveform[:, 0],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=center,
                window=window,
            )
            right = librosa.stft(
                waveform[:, 1],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=center,
                window=window,
            )
            return np.stack([left, right], axis=0)

        raise ValueError("Stereo waveform must have shape [2, N] or [N, 2].")

    raise ValueError("waveform must have shape [N] (mono), [2, N] or [N, 2] (stereo).")


def split_magnitude_phase(stft_complex: np.ndarray):
    """
    Split complex STFT into magnitude and phase components.

    Args:
        stft_complex: Complex STFT array, shape [F, T] or [2, F, T]

    Returns:
        Tuple of (magnitude, phase)
        - magnitude: Non-negative real values (same shape as input)
        - phase: Values in range [-pi, pi] (same shape as input)
    """
    #########################
    # Change by Ryan:
    # Reason:
    # Fail fast with clear error messages when a caller passes an unexpected type/shape.
    # What it does:
    # Validates that the input is a numpy array and has shape [N], [F, T], or [2, F, T].
    if not isinstance(stft_complex, np.ndarray):
        raise TypeError("stft_complex must be a numpy.ndarray.")
    if stft_complex.ndim not in (1, 2, 3):
        raise ValueError("stft_complex must have shape [N], [F, T] (mono), or [2, F, T] (stereo).")
    #########################

    magnitude = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    return magnitude, phase


def normalize_spectrogram(magnitude: np.ndarray, method: str = "minmax"):
    """
    Normalize spectrogram magnitude values.

    Args:
        magnitude: magnitude spectrogram, shape [F, T] or [2, F, T]
        method: "minmax" (scale to [0, 1]), "freq_minmax" (per-frequency scaling), or "none"

    Returns:
        Tuple of (normalized_magnitude, params)
        params contains everything needed to reverse normalization
    """
    #########################
    # Change by Ryan:
    # Reason:
    # Normalize method strings and validate inputs to avoid silent mismatch from
    # whitespace/casing or unexpected shapes. Also keeps min/max params serializable.
    # What it does:
    # - Validates magnitude is a numpy array with expected dimensionality.
    # - Validates method is a non-empty string.
    # - Normalizes method via strip().lower().
    if not isinstance(magnitude, np.ndarray):
        raise TypeError("magnitude must be a numpy.ndarray.")
    if magnitude.ndim not in (2, 3):
        raise ValueError("magnitude must have shape [F, T] or [2, F, T].")
    if not isinstance(method, str) or method.strip() == "":
        raise ValueError("method must be a non-empty string.")

    method = method.strip().lower()
    #########################

    normalized = magnitude.copy()

    if method == "none":
        return normalized, {"method": "none"}

    if method == "minmax":
        #########################
        # Change by Ryan:
        # Reason:
        # Ensure returned min/max params are plain floats and comparisons are stable.
        # What it does:
        # Casts min/max to float and uses an explicit 0.0 comparison.
        min_val = float(magnitude.min())
        max_val = float(magnitude.max())

        if max_val - min_val == 0.0:
            return normalized, {"method": "minmax", "min": min_val, "max": max_val}
        #########################

        normalized = (magnitude - min_val) / (max_val - min_val)
        return normalized, {"method": "minmax", "min": min_val, "max": max_val}

    if method == "freq_minmax":
        if magnitude.ndim != 2:
            raise ValueError("freq_minmax expects magnitude with shape [F, T].")

        f_min = magnitude.min(axis=1, keepdims=True)  # [F, 1]
        f_max = magnitude.max(axis=1, keepdims=True)  # [F, 1]

        denom = np.maximum(f_max - f_min, _FREQ_MINMAX_EPS)
        normalized = np.clip((magnitude - f_min) / denom, 0.0, 1.0)
        return normalized, {"method": "freq_minmax", "f_min": f_min, "f_max": f_max}

    raise ValueError(f"Unknown normalization method: {method}")
