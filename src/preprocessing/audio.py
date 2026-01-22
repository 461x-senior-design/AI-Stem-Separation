# audio.py
# Perorm time-domain audio operations

import numpy as np
import librosa
from pathlib import Path
from typing import Union

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
