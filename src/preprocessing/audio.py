# audio.py
# Perorm time-domain audio operations

from pathlib import Path
from typing import Union

import librosa
import numpy as np

from .constants import TARGET_SAMPLE_RATE


def load_audio(
    path: Union[str, Path], sr: int = TARGET_SAMPLE_RATE, mono: bool = False
) -> tuple[np.ndarray, int]:
    """
    Load audio file with resampling.

    Args:
        path: Path to audio file (.wav, .mp3, .flac)
        sr: Target sample rate (default: 44100)
        mono: If True, convert to mono. If False, keep stereo.

    Returns:
        Tuple of (waveform, sample_rate)
        - waveform: shape (N,) if mono, (2, N) if stereo
        - sample_rate: always equals the sr parameter

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file format not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa.load returns (waveform, sample_rate)
    # mono=False keeps stereo as (2, N), mono=True averages to (N,)
    waveform, loaded_sr = librosa.load(path, sr=sr, mono=mono)

    return waveform, sr


def ensure_stereo(waveform: np.ndarray) -> np.ndarray:
    """
    Ensure waveform is is stereo (2 channels).

    Args:
        waveform: Shape (N,) for mono or (2, N) for stereo

    Returns:
        Stereo waveform with shape (2, N)
        If input is mono, duplicates to both channels.
    """
    if waveform.ndim == 1:
        # Mono: duplicate to both channels
        return np.stack([waveform, waveform])

    if waveform.ndim == 2 and waveform.shape[0] == 2:
        # Already stereo
        return waveform

    raise ValueError(f"Unexpected waveform shape {waveform.shape}")


def normalize_waveform(waveform: np.ndarray, method: str = "peak") -> tuple[np.ndarray, dict]:
    """
    Normalize waveform amplitude.

    For stereo: uses SAME scale factor for both chennels to preserve stereo balance.

    Args:
        waveform: Input waveform [2, N] stereo or [N] mono
        method: "peak" (scale max to 1.0), "rms", or "none"

    Returns:
        Tuple of (normalized_waveform, params)
        params contains everything needed to reverse normalization
    """
    # Because numpy arrays are mutable, we must make a copy.
    # Otherwise, and modifications to normalized would affect waveform
    normalized = waveform.copy()

    if method == "none":
        return normalized, {"method": "none", "scale_factor": 1.0}

    if method == "peak":
        peak = np.abs(waveform).max()

        if peak == 0.0:
            return normalized, {"method": "peak", "scale_factor": 1.0}

        normalized = waveform / peak
        return normalized, {"method": "peak", "scale_factor": peak}

    if method == "rms":
        rms = np.sqrt(np.mean(waveform**2))

        if rms == 0.0:
            return normalized, {"method": "rms", "scale_factor": 1.0}

        normalized = waveform / rms
        return normalized, {"method": "rms", "scale_factor": rms}

    raise ValueError(f"Unkown normalization method: {method}")
