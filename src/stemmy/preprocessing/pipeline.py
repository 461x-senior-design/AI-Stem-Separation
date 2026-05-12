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
)

from .audio import ensure_stereo, load_audio, normalize_waveform
from .spectral import compute_stft, normalize_spectrogram, split_magnitude_phase
from .utility.audio_file_validator import AudioFileValidator, AudioValidationException


@dataclass
class PreprocessingMetadata:
    """
    Stores all info needed to reconstruct audio after separation.

    Notes on sample rates:
      - original_sr: The sample rate returned by load_audio(). In this pipeline, load_audio()
        is called with sr=self.sample_rate, so audio is resampled on load and original_sr will
        typically equal processed_sr. (The native file sample rate is not preserved.)
      - processed_sr: The target sample rate used throughout preprocessing/inference.

    We preserve phase (split_magnitude_phase) for ISTFT reconstruction.
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
        target_channels: int = TARGET_CHANNELS,
        audio_channels: int = TARGET_CHANNELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        center: bool = STFT_CENTER,
        waveform_norm: str = DEFAULT_WAVEFORM_NORM,
        spectrogram_norm: str = DEFAULT_SPECTROGRAM_NORM,
    ):
        self.sample_rate = sample_rate
        self.target_channels = target_channels
        self.audio_channels = audio_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.waveform_norm = waveform_norm
        self.spectrogram_norm = spectrogram_norm

        if int(self.target_channels) != int(TARGET_CHANNELS):
            raise ValueError(
                "target_channels must be %d, got %d."
                % (int(TARGET_CHANNELS), int(self.target_channels))
            )
        if int(self.audio_channels) != int(TARGET_CHANNELS):
            raise ValueError(
                "audio_channels must be %d, got %d."
                % (int(TARGET_CHANNELS), int(self.audio_channels))
            )

    def process(self, audio_path: Union[str, Path]) -> tuple[torch.Tensor, PreprocessingMetadata]:
        """
        Run full preprocessing pipeline.

        Args:
            audio_path Path to input audio file

        Returns:
            Tuple of (tensor, metadata)
            tensor: [1, 2, F, T] normalized stereo magnitude
        """
        audio_path = Path(audio_path)

        validator = AudioFileValidator(str(audio_path))
        is_valid, message = validator.validate()
        if not is_valid:
            raise AudioValidationException(message)

        waveform, sr = load_audio(audio_path, sr=self.sample_rate)
        original_length = waveform.shape[-1]

        waveform = ensure_stereo(waveform)
        if waveform.ndim != 2 or waveform.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "Expected stereo waveform with shape [%d, N], got %s."
                % (int(TARGET_CHANNELS), str(waveform.shape))
            )

        mono_waveform = waveform.mean(axis=0)
        _mono_norm, waveform_params = normalize_waveform(mono_waveform, method=self.waveform_norm)

        scale = float(waveform_params.get("scale_factor", 1.0))
        if scale == 0.0:
            scale = 1.0
        waveform = waveform / scale
        processed_length = waveform.shape[-1]

        stft = compute_stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
        )

        magnitude, phase = split_magnitude_phase(stft)

        mix_magnitude = magnitude
        if mix_magnitude.ndim != 3 or mix_magnitude.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "Unexpected magnitude shape %s. Expected [%d, F, T]."
                % (str(mix_magnitude.shape), int(TARGET_CHANNELS))
            )

        normalized_magnitude, spec_params = normalize_spectrogram(
            mix_magnitude,
            method=self.spectrogram_norm,
        )

        if normalized_magnitude.ndim != 3 or normalized_magnitude.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "Unexpected normalized magnitude shape %s. Expected [%d, F, T]."
                % (str(normalized_magnitude.shape), int(TARGET_CHANNELS))
            )

        tensor = torch.from_numpy(normalized_magnitude).float().unsqueeze(0)

        metadata = PreprocessingMetadata(
            original_path=str(audio_path),
            original_sr=sr,
            processed_sr=self.sample_rate,
            original_length=original_length,
            processed_length=processed_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            n_frames=int(normalized_magnitude.shape[-1]),
            phase=phase,
            waveform_norm_params=waveform_params,
            spectrogram_norm_params=spec_params,
            mix_magnitude=mix_magnitude,
        )

        return tensor, metadata
