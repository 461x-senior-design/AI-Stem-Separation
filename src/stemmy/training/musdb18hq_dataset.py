# src/stemmy/training/musdb18hq_dataset.py
"""MUSDB18-HQ dataset for training a spectrogram-mask U-Net.

This dataset:
- loads aligned mixture + stem audio from the MUSDB18-HQ folder layout
- extracts a fixed-length mono waveform segment
- computes mono STFT magnitude (using StftConfig, including `center`)
- computes per-stem ratio masks as:
      target_s = stem_mag_s / (sum_s(stem_mag_s) + eps)
  which yields a per-(F,T) distribution over stems (sum-to-one across stems)
- optionally normalizes the mixture magnitude spectrogram for model input

Returned tensors:
- mix_norm:     [1, F, T] float32 normalized mixture magnitude
- targets_norm: [S, F, T] float32 per-stem ratio masks (S=4), sum-to-one across stems
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch

from stemmy.constants import (
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_WAVEFORM_NORM,
    STEMS_4,
)
from stemmy.preprocessing import normalize_waveform
from stemmy.training.stft import StftConfig, freq_minmax_normalize, stft_mag_phase_mono


@dataclass(frozen=True)
class CropConfig:
    """Fixed-size STFT crop configuration.

    time_frames determines the number of STFT frames T returned by the dataset.
    """

    time_frames: int = 256  # T


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
        seed: int = 0,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: MUSDB18-HQ root directory containing train/ and test/
            split: "train" or "test"
            stft_cfg: STFT configuration (including center)
            crop_cfg: Crop configuration (fixed number of time frames)
            stems: Optional subset of stems, defaults to STEMS_4
            max_tracks: If set, limit number of tracks in this split
            deterministic: If True, choose a centered crop for each track
            waveform_norm: Waveform normalization method ("peak", "rms", "none")
            spectrogram_norm: Spectrogram normalization method (expected "freq_minmax" or "none")
            seed: Random seed for selecting crop start positions
        """
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
        self.waveform_norm = waveform_norm

        if not isinstance(spectrogram_norm, str) or spectrogram_norm.strip() == "":
            raise ValueError("spectrogram_norm must be a non-empty string.")
        self.spectrogram_norm = spectrogram_norm.strip().lower()

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
        """Compute waveform samples required to yield exactly T STFT frames.

        With torch.stft:
        - center=False:
            T = 1 + floor((N - n_fft) / hop)  => choose N = n_fft + hop*(T-1)
        - center=True:
            padding is n_fft//2 on both sides, effective length N + n_fft
            T = 1 + floor((N + n_fft - n_fft) / hop) = 1 + floor(N / hop)
            => choose N = hop*(T-1)

        Returns:
            Required number of samples N for a crop producing crop_cfg.time_frames frames.
        """
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

    def _read_mono_segment(self, wav_path: Path, start_frame: int, num_frames: int) -> torch.Tensor:
        """Read a mono segment (downmixed) from a wav file.

        Args:
            wav_path: Path to the wav file
            start_frame: Starting frame offset (samples) within the file
            num_frames: Number of frames (samples) to read

        Returns:
            Mono waveform tensor [num_frames] float32
        """
        if not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("start_frame must be a non-negative int.")
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames must be a positive int.")

        info = sf.info(str(wav_path))
        expected_sr = int(self.stft_cfg.sample_rate)
        if int(info.samplerate) != expected_sr:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: got {info.samplerate}, expected "
                f"{expected_sr}"
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

        mono = audio.mean(axis=1)
        mono_t = torch.from_numpy(mono).to(torch.float32)

        if int(mono_t.numel()) != int(num_frames):
            raise RuntimeError(
                f"Short read for {wav_path}: expected {num_frames}, got {mono_t.numel()}"
            )

        return mono_t

    def _dtype_eps(self, t: torch.Tensor) -> float:
        """Get a small epsilon appropriate for the tensor dtype."""
        if not torch.is_floating_point(t):
            return 1e-8
        return float(torch.finfo(t.dtype).eps)

    def _validate_spectrogram_norm(self) -> None:
        """Validate spectrogram normalization mode for this dataset path."""
        if self.spectrogram_norm not in ["freq_minmax", "none"]:
            raise ValueError(
                "Unsupported spectrogram_norm '%s'. Expected 'freq_minmax' or 'none'."
                % (self.spectrogram_norm,)
            )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training example.

        Args:
            idx: Track index

        Returns:
            Tuple of:
              mix_norm: [1, F, T] float32
              targets_norm: [S, F, T] float32
        """
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

        mix_wav = self._read_mono_segment(mix_path, start, self.segment_samples)

        mix_wav_np = mix_wav.detach().cpu().numpy()
        mix_wav_np, mix_norm_params = normalize_waveform(mix_wav_np, method=self.waveform_norm)
        mix_scale = float(mix_norm_params.get("scale_factor", 1.0))
        mix_wav = torch.from_numpy(mix_wav_np).to(torch.float32)

        mix_mag, _mix_phase = stft_mag_phase_mono(mix_wav, self.stft_cfg)
        if int(mix_mag.shape[1]) != int(self.crop_cfg.time_frames):
            raise RuntimeError(
                f"Unexpected time frames: got {mix_mag.shape[1]}, expected "
                f"{self.crop_cfg.time_frames}"
            )

        if self.spectrogram_norm == "none":
            mix_norm = mix_mag
        else:
            mix_norm, _f_min, _f_max = freq_minmax_normalize(mix_mag)

        stem_mags: List[torch.Tensor] = []
        for stem in self.stems:
            stem_path = track_dir / f"{stem}.wav"
            stem_wav = self._read_mono_segment(stem_path, start, self.segment_samples)

            if mix_scale != 0.0:
                stem_wav = stem_wav / mix_scale

            stem_mag, _ = stft_mag_phase_mono(stem_wav, self.stft_cfg)
            stem_mags.append(stem_mag)

        denom = torch.zeros_like(stem_mags[0])
        for sm in stem_mags:
            denom = denom + sm

        eps = max(self._dtype_eps(denom), 1e-8)
        denom_safe = denom + float(eps)

        target_masks: List[torch.Tensor] = []
        for sm in stem_mags:
            target_masks.append(sm / denom_safe)

        mix_norm = mix_norm.unsqueeze(0)  # [1, F, T]
        targets_norm = torch.stack(target_masks, dim=0)  # [S, F, T]

        targets_sum = targets_norm.sum(dim=0, keepdim=False)
        if not torch.isfinite(targets_sum).all():
            raise RuntimeError("Non-finite target masks encountered.")

        return mix_norm, targets_norm
