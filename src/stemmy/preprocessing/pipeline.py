# src/stemmy/preprocessing/pipeline.py
# Orchestrator for the preprocessing modules.

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

#########################
# Change by Ryan:
# Reason:
# Import shared defaults from the project-wide constants module so preprocessing
# stays aligned with training/inference.
from stemmy.constants import (
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
)

#########################
from .audio import ensure_stereo, load_audio, normalize_waveform
from .spectral import compute_stft, normalize_spectrogram, split_magnitude_phase
from .utility.audio_file_validator import AudioFileValidator, AudioValidationException


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
    #########################
    # Change by Ryan:
    # Reason:
    # Persist STFT framing fields and stereo mixture magnitude so downstream
    # reconstruction uses the same framing and can apply mono masks to both channels.
    win_length: int
    center: bool
    #########################
    n_frames: int
    phase: np.ndarray
    waveform_norm_params: dict
    spectrogram_norm_params: dict
    #########################
    # Change by Ryan:
    # Reason:
    # Preserve stereo mixture magnitude for later reconstruction.
    mix_magnitude: np.ndarray
    #########################


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
        #########################
        # Change by Ryan:
        # Reason:
        # Carry framing fields + default norms from centralized constants to keep
        # preprocessing aligned with training/inference.
        win_length: int = WIN_LENGTH,
        center: bool = STFT_CENTER,
        waveform_norm: str = DEFAULT_WAVEFORM_NORM,
        spectrogram_norm: str = DEFAULT_SPECTROGRAM_NORM,
        #########################
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        #########################
        # Change by Ryan:
        # Reason:
        # Persist framing fields for compute_stft and metadata.
        self.win_length = win_length
        self.center = center
        #########################
        self.waveform_norm = waveform_norm
        self.spectrogram_norm = spectrogram_norm

    def process(self, audio_path: Union[str, Path]) -> tuple[torch.Tensor, PreprocessingMetadata]:
        """
        Run full preprocessing pipeline.

        Args:
            audio_path Path to input audio file

        Returns:
            Tuple of (tensor, metadata)
            tensor: [1, 1, F, T] normalized magnitude
        """
        audio_path = Path(audio_path)

        # Use Haedon's validation system
        validator = AudioFileValidator(str(audio_path))
        is_valid, message = validator.validate()
        if not is_valid:
            raise AudioValidationException(message)

        # Load audio
        waveform, sr = load_audio(audio_path, sr=self.sample_rate)
        original_length = waveform.shape[-1]

        # Ensure Stereo
        waveform = ensure_stereo(waveform)

        # Normalize waveform
        waveform, waveform_params = normalize_waveform(waveform, method=self.waveform_norm)
        processed_length = waveform.shape[-1]

        # Compute STFT
        #########################
        # Change by Ryan:
        # Reason:
        # Pass win_length/center explicitly so framing matches training/inference.
        stft = compute_stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
        )
        #########################

        # Split magnitude and phase
        magnitude, phase = split_magnitude_phase(stft)

        #########################
        # Change by Ryan:
        # Reason:
        # Preserve stereo mixture magnitude for reconstruction, but feed mono magnitude into the
        # model.
        mix_magnitude = magnitude
        if mix_magnitude.ndim == 3 and mix_magnitude.shape[0] == 2:
            mono_magnitude = mix_magnitude.mean(axis=0)
        elif mix_magnitude.ndim == 2:
            mono_magnitude = mix_magnitude
        else:
            raise ValueError(f"Unexpected magnitude shape {mix_magnitude.shape}")
        #########################

        # Normalize spectrogram
        #########################
        # Change by Ryan:
        # Reason:
        # Normalize the mono magnitude that is used as model input.
        mono_magnitude, spec_params = normalize_spectrogram(
            mono_magnitude, method=self.spectrogram_norm
        )
        #########################

        # Convert to tensor
        #########################
        # Change by Ryan:
        # Reason:
        # Model input is mono magnitude with shape [1, 1, F, T].
        tensor = torch.from_numpy(mono_magnitude).float().unsqueeze(0).unsqueeze(0)
        #########################

        # Build metadata
        metadata = PreprocessingMetadata(
            original_path=str(audio_path),
            original_sr=sr,
            processed_sr=self.sample_rate,
            original_length=original_length,
            processed_length=processed_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            #########################
            # Change by Ryan:
            # Reason:
            # Persist framing fields used for downstream reconstruction.
            win_length=self.win_length,
            center=self.center,
            #########################
            n_frames=int(mono_magnitude.shape[-1]),
            phase=phase,
            waveform_norm_params=waveform_params,
            spectrogram_norm_params=spec_params,
            #########################
            # Change by Ryan:
            # Reason:
            # Provide stereo mixture magnitude for postprocessing reconstruction.
            mix_magnitude=mix_magnitude,
            #########################
        )

        return tensor, metadata
