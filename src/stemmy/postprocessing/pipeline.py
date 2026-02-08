# pipeline.py
# Ties together all of the postprocessing modules.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from stemmy.preprocessing.pipeline import PreprocessingMetadata

from .audio import denormalize_waveform, export_audio
from .spectral import apply_mask, combine_magnitude_phase, compute_istft, denormalize_spectrogram


@dataclass
class SeparationResult:
    """
    Result of audio separation.

    Contains paths to exported files and/or raw waveforms.
    """

    vocals_path: Optional[Path]
    instrumentals_path: Optional[Path]
    vocals_waveform: Optional[np.ndarray]
    instrumentals_waveform: Optional[np.ndarray]
    sample_rate: int


class Postprocessor:
    """
    Main postprocessing pipeline.

    Example:
        post = Postprocessor()
        result = post.process(model_output, metadata, input_tensor, output_dir, "song")
    """

    def __init__(self):
        pass  # No configuration needed for now

    def process(
        self,
        model_output: torch.Tensor,
        metadata: PreprocessingMetadata,
        input_tensor: torch.Tensor,
        output_dir: Union[str, Path],
        stem_name: str,
        export_files: bool = True,
    ) -> SeparationResult:
        """
        Run full postprocessing pipeline.

        Args:
            model_output: Model predictions, shape [1, 2, F, T]
                          Channel 0 = vocals mask, Channel 1 = instrumentals mask
            metadata: PreprocessingMetadata from preprocessing
            input_tensor: Original preprocessed tensor [1, 2, F, T] (normalized magnitude)
            output_dir: Directory to write output files
            stem_name: Base name for output files (e.g., "song" -> "song_vocals.wav")
            export_files: Whether to write WAV files

        Returns:
            SeparationResult with paths and/or waveforms
        """
        output_dir = Path(output_dir)

        # Recover original magnitude from normalized tensor
        normalized_magnitude = input_tensor.squeeze(0).numpy()  # [2, F, T]
        original_magnitude = denormalize_spectrogram(
            normalized_magnitude, metadata.spectrogram_norm_params
        )

        # Extract masks from model output [1, 2, F, T] -> [F, T] each
        masks = model_output.squeeze(0).numpy()  # [2, F, T]
        vocals_mask = masks[0]  # [F, T]
        instrumentals_mask = masks[1]  # [F, T]

        # Broadcast mono masks to stereo shape [2, F, T]
        vocals_mask = np.stack([vocals_mask, vocals_mask])
        instrumentals_mask = np.stack([instrumentals_mask, instrumentals_mask])

        # Reconstruct each stem
        vocals_waveform = self._reconstruct_stem(vocals_mask, original_magnitude, metadata)
        instrumentals_waveform = self._reconstruct_stem(
            instrumentals_mask, original_magnitude, metadata
        )

        # Export files
        vocals_path = None
        instrumentals_path = None

        if export_files:
            vocals_path = export_audio(
                vocals_waveform,
                output_dir / f"{stem_name}_vocals.wav",
                metadata.processed_sr,
            )
            instrumentals_path = export_audio(
                instrumentals_waveform,
                output_dir / f"{stem_name}_instrumentals.wav",
                metadata.processed_sr,
            )

        return SeparationResult(
            vocals_path=vocals_path,
            instrumentals_path=instrumentals_path,
            vocals_waveform=vocals_waveform,
            instrumentals_waveform=instrumentals_waveform,
            sample_rate=metadata.processed_sr,
        )

    def _reconstruct_stem(
        self, mask: np.ndarray, original_magnitude: np.ndarray, metadata: PreprocessingMetadata
    ) -> np.ndarray:
        """
        Reconstruct a single stem from mask and original magnitude.

        Args:
            mask: Soft mask, shape [2, F, T]
            original_magnitude: Original magnitude spectrogram [2, F, T]
            metadata: PreprocessingMetadata containing phase and norm params

        Returns:
            Waveform, shape [2, N]
        """
        # Apply mask to original magnitude
        masked_magnitude = apply_mask(original_magnitude, mask)

        # Recombine with phase
        stft_complex = combine_magnitude_phase(masked_magnitude, metadata.phase)

        # ISTFT to get waveform
        waveform = compute_istft(
            stft_complex,
            hop_length=metadata.hop_length,
            win_length=metadata.n_fft,
            length=metadata.processed_length,
        )

        # Denormalize waveform
        waveform = denormalize_waveform(waveform, metadata.waveform_norm_params)

        return waveform
