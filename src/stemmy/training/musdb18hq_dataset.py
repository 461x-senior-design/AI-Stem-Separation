# src/stemmy/training/musdb18hq_dataset.py
"""MUSDB18-HQ dataset for training a stereo spectrogram-mask U-Net.

This dataset:
- loads aligned mixture + stem audio from the MUSDB18-HQ folder layout
- extracts a fixed-length stereo waveform segment
- computes stereo STFT magnitude using StftConfig, including `center`
- computes per-stem training masks using one of two modes:
      sum_to_one:
          target_s = stem_mag_s / (sum_s(stem_mags_s) + eps)
      mix_ratio:
          target_s = clamp(stem_mag_s / (mix_mag + eps), 0, 1)
- optionally normalizes the stereo mixture magnitude spectrogram for model input

Returned tensors:
- mix_norm:     [2, F, T] float32 normalized stereo mixture magnitude
- targets_norm: [S, 2, F, T] float32 per-stem stereo target masks
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch

from stemmy.constants import (
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_TARGET_MODE,
    DEFAULT_WAVEFORM_NORM,
    STEMS_4,
    SUPPORTED_TARGET_MODES,
    TARGET_CHANNELS,
)
from stemmy.training.stft import StftConfig, freq_minmax_normalize, stft_mag_phase_stereo


@dataclass(frozen=True)
class CropConfig:
    """Fixed-size STFT crop configuration."""

    time_frames: int = 256


class Musdb18HQDataset(torch.utils.data.Dataset):
    """Dataset for MUSDB18-HQ stems.

    Expects MUSDB18-HQ layout:
      <root>/
        train/<Track Name>/{mixture.wav, drums.wav, bass.wav, vocals.wav, other.wav}
        test/<Track Name>/{...}
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        stft_cfg: StftConfig,
        crop_cfg: CropConfig,
        stems: Optional[List[str]] = None,
        max_tracks: Optional[int] = None,
        deterministic: bool = False,
        waveform_norm: str = DEFAULT_WAVEFORM_NORM,
        spectrogram_norm: str = DEFAULT_SPECTROGRAM_NORM,
        target_mode: str = DEFAULT_TARGET_MODE,
        seed: int = 0,
    ) -> None:
        """Initialize the dataset."""
        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'.")

        if not isinstance(root_dir, str) or root_dir.strip() == "":
            raise ValueError("root_dir must be a non-empty string.")

        if not isinstance(crop_cfg, CropConfig):
            raise ValueError("crop_cfg must be a CropConfig.")

        stft_cfg.validate()

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = split
        self.stft_cfg = stft_cfg
        self.crop_cfg = crop_cfg

        self.stems = list(stems) if stems is not None else list(STEMS_4)
        if len(self.stems) == 0:
            raise ValueError("stems cannot be empty.")

        for stem in self.stems:
            if stem not in STEMS_4:
                raise ValueError(f"Unsupported stem '{stem}'. Expected one of {STEMS_4}.")

        self.deterministic = bool(deterministic)

        if not isinstance(waveform_norm, str) or waveform_norm.strip() == "":
            raise ValueError("waveform_norm must be a non-empty string.")
        self.waveform_norm = waveform_norm.strip().lower()

        if not isinstance(spectrogram_norm, str) or spectrogram_norm.strip() == "":
            raise ValueError("spectrogram_norm must be a non-empty string.")
        self.spectrogram_norm = spectrogram_norm.strip().lower()

        if not isinstance(target_mode, str) or target_mode.strip() == "":
            raise ValueError("target_mode must be a non-empty string.")
        self.target_mode = target_mode.strip().lower()
        if self.target_mode not in SUPPORTED_TARGET_MODES:
            raise ValueError(
                "Unsupported target_mode '%s'. Expected one of %s."
                % (self.target_mode, SUPPORTED_TARGET_MODES)
            )

        self.rng = random.Random(int(seed))

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        track_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        track_dirs.sort(key=lambda p: p.name.lower())

        if max_tracks is not None:
            if not isinstance(max_tracks, int) or max_tracks <= 0:
                raise ValueError("max_tracks must be a positive int.")
            track_dirs = track_dirs[:max_tracks]

        self.track_dirs = track_dirs
        if len(self.track_dirs) == 0:
            raise RuntimeError(f"No track directories found under: {split_dir}")

        self.segment_samples = self._segment_samples_required()
        self._validate_track_files()

    def _segment_samples_required(self) -> int:
        """Compute waveform samples required to yield exactly T STFT frames."""
        t_frames = int(self.crop_cfg.time_frames)
        n_fft = int(self.stft_cfg.n_fft)
        hop = int(self.stft_cfg.hop_length)

        if t_frames <= 1:
            raise ValueError("time_frames must be >= 2.")
        if n_fft <= 0 or hop <= 0:
            raise ValueError("Invalid STFT config: n_fft and hop_length must be > 0.")

        if bool(self.stft_cfg.center):
            return hop * (t_frames - 1)

        return n_fft + hop * (t_frames - 1)

    def _validate_track_files(self) -> None:
        """Validate that every track has mixture.wav and each requested stem wav."""
        required = ["mixture.wav"] + [f"{stem}.wav" for stem in self.stems]
        missing: List[str] = []

        for track_dir in self.track_dirs:
            for filename in required:
                if not (track_dir / filename).exists():
                    missing.append(str(track_dir / filename))

        if missing:
            msg = "Missing required files:\n" + "\n".join(missing[:50])
            if len(missing) > 50:
                msg += f"\n... and {len(missing) - 50} more"
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        """Return number of tracks in this dataset split."""
        return len(self.track_dirs)

    def _read_stereo_segment(
        self,
        wav_path: Path,
        start_frame: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Read a stereo segment from a wav file.

        Returns:
            Tensor [2, N]
        """
        if not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("start_frame must be a non-negative int.")
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames must be a positive int.")

        info = sf.info(str(wav_path))
        expected_sr = int(self.stft_cfg.sample_rate)
        if int(info.samplerate) != expected_sr:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: "
                f"got {info.samplerate}, expected {expected_sr}"
            )

        audio, sr = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
            frames=num_frames,
            start=start_frame,
        )
        if int(sr) != expected_sr:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: got {sr}, expected {expected_sr}"
            )

        if int(audio.shape[1]) != int(TARGET_CHANNELS):
            raise ValueError(
                f"Channel count mismatch for {wav_path}: "
                f"got {audio.shape[1]}, expected {TARGET_CHANNELS}"
            )

        stereo = torch.from_numpy(audio.T).to(torch.float32)

        if int(stereo.shape[1]) != int(num_frames):
            raise RuntimeError(
                f"Short read for {wav_path}: expected {num_frames}, got {stereo.shape[1]}"
            )

        return stereo

    def _normalize_stereo_waveform(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Normalize a stereo waveform with one shared scale across both channels."""
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [C, N].")
        if waveform.shape[0] != TARGET_CHANNELS:
            raise ValueError(
                "waveform must have %d channels, got %d."
                % (int(TARGET_CHANNELS), int(waveform.shape[0]))
            )

        method = self.waveform_norm
        if method == "none":
            return waveform, 1.0

        if method == "peak":
            scale = float(waveform.abs().max().item())
        elif method == "rms":
            scale = float(torch.sqrt(torch.mean(waveform * waveform)).item())
        else:
            raise ValueError(
                "Unsupported waveform_norm '%s'. Expected 'peak', 'rms', or 'none'." % method
            )

        if scale <= 0.0:
            return waveform, 1.0

        return waveform / scale, scale

    def _dtype_eps(self, tensor: torch.Tensor) -> float:
        """Get a small epsilon appropriate for the tensor dtype."""
        if not torch.is_floating_point(tensor):
            return 1e-8
        return float(torch.finfo(tensor.dtype).eps)

    def _validate_spectrogram_norm(self) -> None:
        """Validate spectrogram normalization mode for this dataset path."""
        if self.spectrogram_norm not in ["freq_minmax", "none"]:
            raise ValueError(
                "Unsupported spectrogram_norm '%s'. Expected 'freq_minmax' or 'none'."
                % (self.spectrogram_norm,)
            )

    def _build_target_masks(
        self,
        mix_mag: torch.Tensor,
        stem_mags: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Build per-stem stereo target masks according to the configured target mode."""
        if len(stem_mags) == 0:
            raise RuntimeError("stem_mags cannot be empty.")

        if self.target_mode == "sum_to_one":
            denom = torch.zeros_like(stem_mags[0])
            for stem_mag in stem_mags:
                denom = denom + stem_mag

            eps = max(self._dtype_eps(denom), 1e-8)
            denom_safe = denom + float(eps)

            target_masks: List[torch.Tensor] = []
            for stem_mag in stem_mags:
                target_masks.append(stem_mag / denom_safe)
            return target_masks

        if self.target_mode == "mix_ratio":
            eps = max(self._dtype_eps(mix_mag), 1e-8)
            denom_safe = mix_mag + float(eps)

            target_masks = []
            for stem_mag in stem_mags:
                target_masks.append(torch.clamp(stem_mag / denom_safe, min=0.0, max=1.0))
            return target_masks

        raise RuntimeError("Unsupported target_mode encountered: %s" % self.target_mode)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example."""
        if not isinstance(idx, int):
            raise IndexError("idx must be an int.")
        if idx < 0 or idx >= len(self.track_dirs):
            raise IndexError("idx out of range")

        self._validate_spectrogram_norm()

        track_dir = self.track_dirs[idx]
        mix_path = track_dir / "mixture.wav"

        info = sf.info(str(mix_path))
        total_frames = int(info.frames)

        if total_frames < self.segment_samples:
            raise ValueError(
                f"Track too short for crop: {mix_path} has {total_frames} frames, "
                f"need {self.segment_samples}"
            )

        if self.deterministic:
            start = (total_frames - self.segment_samples) // 2
        else:
            start = self.rng.randint(0, total_frames - self.segment_samples)

        mix_wav = self._read_stereo_segment(mix_path, start, self.segment_samples)
        mix_wav, mix_scale = self._normalize_stereo_waveform(mix_wav)

        mix_mag, _mix_phase = stft_mag_phase_stereo(mix_wav, self.stft_cfg)
        if int(mix_mag.shape[2]) != int(self.crop_cfg.time_frames):
            raise RuntimeError(
                f"Unexpected time frames: got {mix_mag.shape[2]}, expected "
                f"{self.crop_cfg.time_frames}"
            )

        if self.spectrogram_norm == "none":
            mix_norm = mix_mag
        else:
            mix_norm, _f_min, _f_max = freq_minmax_normalize(mix_mag)

        stem_mags: List[torch.Tensor] = []
        for stem in self.stems:
            stem_path = track_dir / f"{stem}.wav"
            stem_wav = self._read_stereo_segment(stem_path, start, self.segment_samples)

            if mix_scale != 0.0:
                stem_wav = stem_wav / mix_scale

            stem_mag, _stem_phase = stft_mag_phase_stereo(stem_wav, self.stft_cfg)
            stem_mags.append(stem_mag)

        target_masks = self._build_target_masks(mix_mag=mix_mag, stem_mags=stem_mags)

        targets_norm = torch.stack(target_masks, dim=0)

        if not torch.isfinite(targets_norm).all():
            raise RuntimeError("Non-finite target masks encountered.")

        return mix_norm, targets_norm
