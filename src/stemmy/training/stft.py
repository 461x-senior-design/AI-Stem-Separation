# src/stemmy/training/stft.py
"""STFT helpers for training.

This module defines:
- StftConfig: a small configuration object used across training code.
- stft_mag_phase_mono: compute mono STFT magnitude and phase using torch.stft.
- freq_minmax_normalize: per-frequency min/max normalization used for model inputs.

The key requirement is that STFT framing is consistent across the project.
In particular, the `center` setting must match stemmy.constants.STFT_CENTER.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from stemmy.constants import HOP_LENGTH, N_FFT, STFT_CENTER, TARGET_SAMPLE_RATE, WIN_LENGTH, WINDOW

logger = logging.getLogger(__name__)

# Module-level window cache: keyed by (win_length, device_str, dtype_str)
_window_cache: dict[tuple[int, str, str], torch.Tensor] = {}


@dataclass(frozen=True)
class StftConfig:
    """Configuration for computing STFTs in training.

    Attributes:
        sample_rate: Sample rate in Hz (stored for completeness; not used by torch.stft)
        n_fft: FFT size (window size in samples)
        hop_length: Hop length in samples
        win_length: Window length in samples
        center: Whether to pad the signal so frames are centered
        window: Window function name
    """

    sample_rate: int = TARGET_SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    center: bool = STFT_CENTER
    window: str = WINDOW

    def validate(self) -> None:
        """Validate configuration values."""
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")
        if not isinstance(self.n_fft, int) or self.n_fft <= 0:
            raise ValueError("n_fft must be > 0.")
        if not isinstance(self.hop_length, int) or self.hop_length <= 0:
            raise ValueError("hop_length must be > 0.")
        if not isinstance(self.win_length, int) or self.win_length <= 0:
            raise ValueError("win_length must be > 0.")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft.")

        if not isinstance(self.window, str) or self.window.strip() == "":
            raise ValueError("window must be a non-empty string.")

        window = self.window.strip().lower()
        expected = str(WINDOW).strip().lower()
        if window != expected:
            raise ValueError("window must match stemmy.constants.WINDOW (%s)." % (str(WINDOW),))

        if window != "hann":
            raise ValueError("Only WINDOW='hann' is supported in this training path.")


def _build_window(cfg: StftConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build (or retrieve cached) a torch window tensor for torch.stft.

    Windows are cached by (win_length, device, dtype) to avoid re-creating them
    on every STFT call. This is safe because Hann windows are deterministic and
    immutable for a given size/device/dtype combination.
    """
    window_name = cfg.window.strip().lower()
    if window_name != "hann":
        raise ValueError(f"Unsupported window type: {cfg.window}")

    cache_key = (cfg.win_length, str(device), str(dtype))
    if cache_key not in _window_cache:
        _window_cache[cache_key] = torch.hann_window(
            cfg.win_length, device=device, dtype=dtype
        )
        logger.debug(
            "Created and cached STFT window: win_length=%d device=%s dtype=%s",
            cfg.win_length, device, dtype,
        )
    return _window_cache[cache_key]


def stft_mag_phase_mono(
    waveform: torch.Tensor,
    cfg: StftConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mono STFT magnitude and phase.

    Args:
        waveform: Mono waveform tensor with shape [N]
        cfg: STFT configuration

    Returns:
        Tuple of (magnitude, phase):
            magnitude: [F, T] float32/float64
            phase: [F, T] float32/float64 (radians)
    """
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("waveform must be a torch.Tensor.")
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1D (mono) with shape [N].")

    cfg.validate()

    device = waveform.device
    dtype = waveform.dtype
    window = _build_window(cfg, device=device, dtype=dtype)

    stft_complex = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=bool(cfg.center),
        return_complex=True,
    )

    magnitude = torch.abs(stft_complex)
    phase = torch.angle(stft_complex)
    return magnitude, phase


def freq_minmax_normalize(
    magnitude: torch.Tensor,
    eps: Optional[float] = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-frequency min/max normalization.

    This normalizes each frequency bin independently across time, mapping
    values to [0, 1] based on per-bin min/max.

    Args:
        magnitude: STFT magnitude with shape [F, T]
        eps: Small value to avoid division by zero. If None, uses torch.finfo(dtype).eps.

    Returns:
        Tuple of (normalized, f_min, f_max):
            normalized: [F, T]
            f_min: [F, 1]
            f_max: [F, 1]
    """
    if not isinstance(magnitude, torch.Tensor):
        raise ValueError("magnitude must be a torch.Tensor.")
    if magnitude.ndim != 2:
        raise ValueError("magnitude must have shape [F, T].")
    if not torch.is_floating_point(magnitude):
        raise ValueError("magnitude must be a floating-point tensor.")

    if eps is None:
        eps_val = float(torch.finfo(magnitude.dtype).eps)
    else:
        try:
            eps_val = float(eps)
        except (TypeError, ValueError):
            raise ValueError("eps must be a float or None.")
        if eps_val <= 0.0:
            raise ValueError("eps must be > 0.")

    f_min = magnitude.min(dim=1, keepdim=True).values
    f_max = magnitude.max(dim=1, keepdim=True).values
    denom = torch.clamp(f_max - f_min, min=eps_val)
    normalized = (magnitude - f_min) / denom
    normalized = torch.clamp(normalized, 0.0, 1.0)
    return normalized, f_min, f_max
