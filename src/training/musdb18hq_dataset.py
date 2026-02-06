import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch

from src.constants import (
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STEMS_4,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
)
from src.preprocessing.audio import normalize_waveform
from src.training.stft import freq_minmax_normalize, stft_mag_phase_mono


@dataclass(frozen=True)
class StftConfig:
    sample_rate: int = TARGET_SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH


@dataclass(frozen=True)
class CropConfig:
    time_frames: int = 256  # T
    # With center=False:
    # N = n_fft + hop*(T-1)
    # seconds = N / sample_rate


class Musdb18HQDataset(torch.utils.data.Dataset):
    """
    Expects MUSDB18-HQ layout:
      <root>/
        train/<Track Name>/{mixture.wav, drums.wav, bass.wav, vocals.wav, other.wav}
        test/<Track Name>/{...}

    Returns per item:
      mix_norm:     [1, F, T] float32
      targets_norm: [S, F, T] float32  (S=4)
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
        seed: int = 0,
    ) -> None:
        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'.")

        self.root_dir = Path(root_dir)
        self.split = split
        self.stft_cfg = stft_cfg
        self.crop_cfg = crop_cfg
        self.stems = stems if stems is not None else STEMS_4

        if len(self.stems) == 0:
            raise ValueError("stems cannot be empty.")

        for s in self.stems:
            if s not in STEMS_4:
                raise ValueError(f"Unsupported stem '{s}'. Expected one of {STEMS_4}.")

        self.deterministic = deterministic
        self.waveform_norm = waveform_norm
        self.rng = random.Random(seed)

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        track_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        track_dirs.sort(key=lambda p: p.name.lower())

        if max_tracks is not None:
            if max_tracks <= 0:
                raise ValueError("max_tracks must be positive.")
            track_dirs = track_dirs[:max_tracks]

        self.track_dirs = track_dirs
        if len(self.track_dirs) == 0:
            raise RuntimeError(f"No track directories found under: {split_dir}")

        self.segment_samples = self._segment_samples_required()

        self._validate_track_files()

    def _segment_samples_required(self) -> int:
        T = self.crop_cfg.time_frames
        n_fft = self.stft_cfg.n_fft
        hop = self.stft_cfg.hop_length
        if T <= 1:
            raise ValueError("time_frames must be >= 2.")
        return n_fft + hop * (T - 1)

    def _validate_track_files(self) -> None:
        required = ["mixture.wav"] + [f"{s}.wav" for s in self.stems]
        bad = []
        for td in self.track_dirs:
            for fn in required:
                if not (td / fn).exists():
                    bad.append(str(td / fn))
        if bad:
            msg = "Missing required files:\n" + "\n".join(bad[:50])
            if len(bad) > 50:
                msg += f"\n... and {len(bad) - 50} more"
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.track_dirs)

    def _read_mono_segment(self, wav_path: Path, start_frame: int, num_frames: int) -> torch.Tensor:
        info = sf.info(str(wav_path))
        if info.samplerate != self.stft_cfg.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: got {info.samplerate}, "
                f"expected {self.stft_cfg.sample_rate}"
            )

        audio, sr = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
            frames=num_frames,
            start=start_frame,
        )
        if sr != self.stft_cfg.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: got {sr}, "
                "expected {self.stft_cfg.sample_rate}"
            )

        mono = audio.mean(axis=1)
        mono_t = torch.from_numpy(mono).to(torch.float32)
        if mono_t.numel() != num_frames:
            raise RuntimeError(
                f"Short read for {wav_path}: expected {num_frames}, got {mono_t.numel()}"
            )
        return mono_t

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.track_dirs):
            raise IndexError("idx out of range")

        td = self.track_dirs[idx]
        mix_path = td / "mixture.wav"

        info = sf.info(str(mix_path))
        total_frames = info.frames
        if total_frames < self.segment_samples:
            raise ValueError(
                f"Track too short for crop: {mix_path} has {total_frames} frames, "
                "need {self.segment_samples}"
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

        mix_mag, _mix_phase = stft_mag_phase_mono(
            mix_wav,
            n_fft=self.stft_cfg.n_fft,
            hop_length=self.stft_cfg.hop_length,
            win_length=self.stft_cfg.win_length,
        )
        if mix_mag.shape[1] != self.crop_cfg.time_frames:
            raise RuntimeError(
                f"Unexpected time frames: got {mix_mag.shape[1]}, "
                "expected {self.crop_cfg.time_frames}"
            )

        mix_norm, _f_min, _f_max = freq_minmax_normalize(mix_mag)

        eps = 1e-8
        mix_mag_safe = mix_mag + eps

        target_masks = []
        for stem in self.stems:
            stem_path = td / f"{stem}.wav"
            stem_wav = self._read_mono_segment(stem_path, start, self.segment_samples)
            if mix_scale != 0.0:
                stem_wav = stem_wav / mix_scale
            stem_mag, _ = stft_mag_phase_mono(
                stem_wav,
                n_fft=self.stft_cfg.n_fft,
                hop_length=self.stft_cfg.hop_length,
                win_length=self.stft_cfg.win_length,
            )

            stem_mask = stem_mag / mix_mag_safe
            stem_mask = torch.clamp(stem_mask, 0.0, 1.0)
            target_masks.append(stem_mask)

        mix_norm = mix_norm.unsqueeze(0)  # [1, F, T]
        targets_norm = torch.stack(target_masks, dim=0)  # [S, F, T]
        targets_norm = targets_norm / (targets_norm.sum(dim=0, keepdim=True) + eps)
        return mix_norm, targets_norm
