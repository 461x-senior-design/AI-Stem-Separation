# postprocessing.py
# Time-domain operations.
# - Denormalize waveform
# - Export audio to WAV file

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf


def denormalize_waveform(waveform: np.ndarray, params: dict) -> np.ndarray:
    """
    Reverse waveform normalization.

    Args:
        waveform: Normalized waveform, shape [2, N]
        params: Parameters from normalize_waveform()

    Returns:
        Waveform at original scale
    """
    method = params.get("method")
    scale_factor = params.get("scale_factor", 1.0)

    if method == "none":
        return waveform.copy()

    # Both peak and rms are reversed by multiplying
    return waveform * scale_factor


def export_audio(
    waveform: np.ndarray, path: Union[str, Path], sample_rate: int, subtype: str = "PCM_16"
) -> Path:
    """
    Export waveform to audio file.

    Args:
        waveform: Audio data, shape [2, N] for stereo
        path: Output file path
        sample_rate: Sample rate in Hz
        subtype: Audio subtype (PCM_16 or PCM_24)

    Returns:
        Path to the written file
    """
    path = Path(path)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clip to [-1, 1] to prevent clipping distortion
    waveform = np.clip(waveform, -1.0, 1.0)

    # soundfile expects [N, channels], we have [channels, N]
    waveform_transposed = waveform.T

    sf.write(path, waveform_transposed, sample_rate, subtype=subtype)

    return path

