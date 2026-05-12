# pipeline.py
# Ties together all of the postprocessing modules.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from stemmy.constants import TARGET_CHANNELS
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
    stem_paths: dict[str, Optional[Path]]
    stem_waveforms: dict[str, np.ndarray]
    sample_rate: int


class Postprocessor:
    """
    Main postprocessing pipeline.

    Example:
        post = Postprocessor()
        result = post.process(model_output, metadata, input_tensor, output_dir, "song")
    """

    def __init__(self):
        pass

    def process(
        self,
        model_output: torch.Tensor,
        metadata: PreprocessingMetadata,
        input_tensor: torch.Tensor,
        output_dir: Union[str, Path],
        stem_name: str,
        export_files: bool = True,
        stems: Optional[list[str]] = None,
    ) -> SeparationResult:
        """
        Run full postprocessing pipeline.

        Args:
            model_output: [1, 2, F, T] when stems is None; [1, S, 2, F, T] when stems is provided
            metadata: PreprocessingMetadata from preprocessing
            input_tensor: Original preprocessed tensor [1, 2, F, T] normalized magnitude
            output_dir: Directory to write output files
            stem_name: Base name for output files (e.g., "song" -> "song_vocals.wav")
            export_files: Whether to write WAV files
            stems: Optional ordered stem names matching model output stem dimension

        Returns:
            SeparationResult with paths and/or waveforms
        """
        output_dir = Path(output_dir)

        if export_files:
            output_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(metadata, "mix_magnitude"):
            original_magnitude = getattr(metadata, "mix_magnitude")
        else:
            normalized_magnitude = input_tensor.squeeze(0).detach().cpu().numpy()
            original_magnitude = denormalize_spectrogram(
                normalized_magnitude, metadata.spectrogram_norm_params
            )

            if original_magnitude.ndim == 3 and original_magnitude.shape[0] == 1:
                original_magnitude = np.concatenate(
                    [original_magnitude, original_magnitude], axis=0
                )
            elif original_magnitude.ndim == 2:
                original_magnitude = np.stack([original_magnitude, original_magnitude])

        if original_magnitude.ndim != 3 or original_magnitude.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "original_magnitude must have shape [%d, F, T], got %s."
                % (int(TARGET_CHANNELS), str(original_magnitude.shape))
            )

        if stems is not None:
            stem_list = [str(s) for s in stems]
            if len(stem_list) == 0:
                raise ValueError("stems cannot be empty.")

            masks = model_output.squeeze(0).detach().cpu().numpy()

            if masks.ndim == 3:
                if masks.shape[0] != len(stem_list):
                    raise ValueError(
                        "Mask stem count (%d) does not match stems (%d)."
                        % (int(masks.shape[0]), int(len(stem_list)))
                    )
                masks = np.stack([masks, masks], axis=1)
            elif masks.ndim == 4:
                if masks.shape[0] != len(stem_list):
                    raise ValueError(
                        "Mask stem count (%d) does not match stems (%d)."
                        % (int(masks.shape[0]), int(len(stem_list)))
                    )
                if masks.shape[1] != TARGET_CHANNELS:
                    raise ValueError(
                        "Mask channel count (%d) does not match expected channels (%d)."
                        % (int(masks.shape[1]), int(TARGET_CHANNELS))
                    )
            else:
                raise ValueError("model_output must have shape [1, S, F, T] or [1, S, 2, F, T].")

            stem_paths: dict[str, Optional[Path]] = {}
            stem_waveforms: dict[str, np.ndarray] = {}

            for stem_idx, stem in enumerate(stem_list):
                stem_mask = masks[stem_idx]
                waveform = self._reconstruct_stem(stem_mask, original_magnitude, metadata)

                if (
                    waveform.ndim == 2
                    and waveform.shape[0] != TARGET_CHANNELS
                    and waveform.shape[1] == TARGET_CHANNELS
                ):
                    waveform = waveform.T
                if waveform.ndim != 2 or waveform.shape[0] != TARGET_CHANNELS:
                    raise ValueError(
                        "Reconstructed waveform has wrong shape %s, expected [%d, N]."
                        % (str(waveform.shape), int(TARGET_CHANNELS))
                    )

                stem_waveforms[stem] = waveform

                if export_files:
                    stem_paths[stem] = export_audio(
                        waveform,
                        output_dir / f"{stem_name}_{stem}.wav",
                        metadata.processed_sr,
                    )

            if not export_files:
                stem_paths = {}

            vocals_path = None
            instrumentals_path = None
            vocals_waveform = None
            instrumentals_waveform = None

            if "vocals" in stem_waveforms:
                vocals_waveform = stem_waveforms["vocals"]
                vocals_path = stem_paths.get("vocals", None)

            if "instrumentals" in stem_waveforms:
                instrumentals_waveform = stem_waveforms["instrumentals"]
                instrumentals_path = stem_paths.get("instrumentals", None)

            return SeparationResult(
                vocals_path=vocals_path,
                instrumentals_path=instrumentals_path,
                vocals_waveform=vocals_waveform,
                instrumentals_waveform=instrumentals_waveform,
                stem_paths=stem_paths,
                stem_waveforms=stem_waveforms,
                sample_rate=metadata.processed_sr,
            )

        masks = model_output.squeeze(0).detach().cpu().numpy()
        if masks.ndim != 3 or masks.shape[0] != 2:
            raise ValueError("model_output must have shape [1, 2, F, T].")

        vocals_mask = masks[0]
        instrumentals_mask = masks[1]

        vocals_mask = np.stack([vocals_mask, vocals_mask])
        instrumentals_mask = np.stack([instrumentals_mask, instrumentals_mask])

        vocals_waveform = self._reconstruct_stem(vocals_mask, original_magnitude, metadata)
        instrumentals_waveform = self._reconstruct_stem(
            instrumentals_mask, original_magnitude, metadata
        )

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

        if export_files:
            stem_paths = {
                "vocals": vocals_path,
                "instrumentals": instrumentals_path,
            }
        else:
            stem_paths = {}

        stem_waveforms = {
            "vocals": vocals_waveform,
            "instrumentals": instrumentals_waveform,
        }

        return SeparationResult(
            vocals_path=vocals_path,
            instrumentals_path=instrumentals_path,
            vocals_waveform=vocals_waveform,
            instrumentals_waveform=instrumentals_waveform,
            stem_paths=stem_paths,
            stem_waveforms=stem_waveforms,
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
        if mask.ndim != 3 or mask.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "mask must have shape [%d, F, T], got %s." % (int(TARGET_CHANNELS), str(mask.shape))
            )
        if original_magnitude.ndim != 3 or original_magnitude.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "original_magnitude must have shape [%d, F, T], got %s."
                % (int(TARGET_CHANNELS), str(original_magnitude.shape))
            )

        masked_magnitude = apply_mask(original_magnitude, mask)

        stft_complex = combine_magnitude_phase(masked_magnitude, metadata.phase)

        win_length = getattr(metadata, "win_length", metadata.n_fft)
        center = getattr(metadata, "center", False)

        waveform = compute_istft(
            stft_complex,
            hop_length=metadata.hop_length,
            win_length=win_length,
            length=metadata.processed_length,
            center=center,
        )

        waveform = denormalize_waveform(waveform, metadata.waveform_norm_params)

        return waveform
