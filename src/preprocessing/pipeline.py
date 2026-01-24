# pipeline.py
# Orchestrator for the preprocessing modules.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import ensure_stereo, load_audio, normalize_waveform
from .constants import HOP_LENGTH, N_FFT, TARGET_SAMPLE_RATE
from .spectral import compute_stft, normalize_spectrogram, split_magnitude_phase


@dataclass
class PreprocessingMetadata:
    """
    Stores all info needed to reconstruct audio after separation.

    We are preserving phase (split_magnitude_phase) for ISTFT reconstruction.
    """

    original_path: str
    original_sr: int
    processed_sr: int
    original_length: int
    processed_length: int
    n_fft: int
    hop_length: int
    n_frames: int
    phase: np.ndarray
    waveform_norm_params: dict
    spectrogram_norm_params: dict


class Preprocessor:
    """
    Main preprocessing pipeline.

    Example:
        prep = Preprocessor()
        tenseor, metadata = prep.process("song.wav")
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        waveform_norm: str = "peak",
        spectrogram_norm: str = "minmax",
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.waveform_norm = waveform_norm
        self.spectrogram_norm = spectrogram_norm

    def process(self, audio_path: str | Path) -> tuple[torch.Tensor, PreprocessingMetadata]:
        """
        Run full preprocessing pipeline.

        Args:
            audio_path Path to input audio file

        Returns:
            Tuple of (tensor, metadata)
            tensor: [1, 1, F T] normalized magnitude
        """
        audio_path = Path(audio_path)

        # Load audio
        waveform, sr = load_audio(audio_path, sr=self.sample_rate)
        original_length = waveform.shape[-1]

        # Ensure Stereo
        waveform = ensure_stereo(waveform)

        # Normalize waveform
        waveform, waveform_params = normalize_waveform(waveform, method=self.waveform_norm)
        processed_length = waveform.shape[-1]

        # Compute STFT
        stft = compute_stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length)

        # Split magnitude and phase
        magnitude, phase = split_magnitude_phase(stft)

        # Normalize spectrogram
        magnitude, spec_params = normalize_spectrogram(magnitude, method=self.spectrogram_norm)

        # Convert to tensor
        tensor = torch.from_numpy(magnitude).float().unsqueeze(0)

        # Build metadata
        metadata = PreprocessingMetadata(
            original_path=str(audio_path),
            original_sr=sr,
            processed_sr=self.sample_rate,
            original_length=original_length,
            processed_length=processed_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_frames=magnitude.shape[-1],
            phase=phase,
            waveform_norm_params=waveform_params,
            spectrogram_norm_params=spec_params,
        )

        return tensor, metadata
