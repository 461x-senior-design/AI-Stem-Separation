# src/stemmy/preprocessing/pipeline.py
# Orchestrator for the preprocessing modules.

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from stemmy.constants import (
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STFT_CENTER,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    WINDOW,
)

from .audio import ensure_stereo, load_audio, normalize_waveform
from .spectral import compute_stft, normalize_spectrogram, split_magnitude_phase
from .utility.audio_file_validator import AudioFileValidator, AudioValidationException


@dataclass
class PreprocessingMetadata:
    """
    Stores all info needed to reconstruct audio after separation.

    Notes on sample rates:
      - original_sr: The sample rate returned by load_audio(). In this pipeline,
        load_audio() is called with sr=self.sample_rate, so audio is resampled on load.
      - processed_sr: The target sample rate used throughout preprocessing/inference.

    We preserve stereo phase for ISTFT reconstruction.
    """

    original_path: str
    original_sr: int
    processed_sr: int
    original_length: int
    processed_length: int
    n_fft: int
    hop_length: int
    win_length: int
    center: bool
    n_frames: int
    phase: np.ndarray
    waveform_norm_params: dict
    spectrogram_norm_params: dict
    mix_magnitude: np.ndarray


class Preprocessor:
    """
    Main preprocessing pipeline.

    Example:
        prep = Preprocessor()
        tensor, metadata = prep.process("song.wav")
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        audio_channels: int = TARGET_CHANNELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        center: bool = STFT_CENTER,
        window: str = WINDOW,
        waveform_norm: str = DEFAULT_WAVEFORM_NORM,
        spectrogram_norm: str = DEFAULT_SPECTROGRAM_NORM,
        eps: float = 1e-8,
    ):
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if not isinstance(audio_channels, int) or audio_channels <= 0:
            raise ValueError("audio_channels must be a positive integer.")
        if not isinstance(n_fft, int) or n_fft <= 0:
            raise ValueError("n_fft must be a positive integer.")
        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError("hop_length must be a positive integer.")
        if not isinstance(win_length, int) or win_length <= 0:
            raise ValueError("win_length must be a positive integer.")
        if win_length > n_fft:
            raise ValueError("win_length must be <= n_fft.")
        if not isinstance(center, bool):
            raise TypeError("center must be a bool.")
        if not isinstance(window, str) or window.strip() == "":
            raise ValueError("window must be a non-empty string.")
        if not isinstance(waveform_norm, str) or waveform_norm.strip() == "":
            raise ValueError("waveform_norm must be a non-empty string.")
        if not isinstance(spectrogram_norm, str) or spectrogram_norm.strip() == "":
            raise ValueError("spectrogram_norm must be a non-empty string.")
        if not isinstance(eps, (float, int)) or float(eps) <= 0.0:
            raise ValueError("eps must be > 0.")

        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.window = window
        self.waveform_norm = waveform_norm
        self.spectrogram_norm = spectrogram_norm
        self.eps = float(eps)

    def process(self, audio_path: Union[str, Path]) -> tuple[torch.Tensor, PreprocessingMetadata]:
        """
        Run full preprocessing pipeline.

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (tensor, metadata)
            tensor: [1, 2, F, T] normalized stereo magnitude
        """
        audio_path = Path(audio_path)

        validator = AudioFileValidator(str(audio_path))
        is_valid, message = validator.validate()
        if not is_valid:
            raise AudioValidationException(message)

        waveform, sr = load_audio(audio_path, sr=self.sample_rate, mono=False)
        original_length = int(waveform.shape[-1])

        waveform = ensure_stereo(waveform)

        if waveform.ndim != 2 or waveform.shape[0] != self.audio_channels:
            raise ValueError(
                "Expected stereo waveform with shape [%d, N], got %s"
                % (int(self.audio_channels), str(waveform.shape))
            )

        waveform, waveform_params = normalize_waveform(waveform, method=self.waveform_norm)
        processed_length = int(waveform.shape[-1])

        stft = compute_stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=self.window,
        )

        magnitude, phase = split_magnitude_phase(stft)

        if magnitude.ndim != 3 or magnitude.shape[0] != self.audio_channels:
            raise ValueError(
                "Expected stereo magnitude with shape [%d, F, T], got %s"
                % (int(self.audio_channels), str(magnitude.shape))
            )

        mix_magnitude = magnitude.copy()

        normalized_magnitude, spec_params = normalize_spectrogram(
            magnitude,
            method=self.spectrogram_norm,
        )

        if normalized_magnitude.ndim != 3 or normalized_magnitude.shape[0] != self.audio_channels:
            raise ValueError(
                "Expected normalized stereo magnitude with shape [%d, F, T], got %s"
                % (int(self.audio_channels), str(normalized_magnitude.shape))
            )

        tensor = torch.from_numpy(normalized_magnitude).float().unsqueeze(0)

        metadata = PreprocessingMetadata(
            original_path=str(audio_path),
            original_sr=int(sr),
            processed_sr=int(self.sample_rate),
            original_length=int(original_length),
            processed_length=int(processed_length),
            n_fft=int(self.n_fft),
            hop_length=int(self.hop_length),
            win_length=int(self.win_length),
            center=bool(self.center),
            n_frames=int(normalized_magnitude.shape[-1]),
            phase=phase,
            waveform_norm_params=waveform_params,
            spectrogram_norm_params=spec_params,
            mix_magnitude=mix_magnitude,
        )

        return tensor, metadata
