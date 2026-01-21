import torch


def stft_mag_phase_mono(
    wav_mono: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    wav_mono: 1D tensor [N] float32
    Returns:
      mag:   [F, T] float32
      phase: [F, T] float32 (angle)
    Notes:
      - center=False so that frame counts are deterministic:
        T = 1 + (N - n_fft) // hop_length
    """
    if wav_mono.ndim != 1:
        raise ValueError("wav_mono must be a 1D tensor [N].")

    if wav_mono.dtype != torch.float32:
        wav_mono = wav_mono.to(torch.float32)

    if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError("n_fft, hop_length, win_length must be positive integers.")

    if win_length > n_fft:
        raise ValueError("win_length must be <= n_fft.")

    window = torch.hann_window(win_length, periodic=True, dtype=torch.float32, device=wav_mono.device)

    spec = torch.stft(
        wav_mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return mag, phase


def istft_from_mag_phase(
    mag: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    length: int,
) -> torch.Tensor:
    """
    mag, phase: [F, T]
    Returns wav_mono: [length]
    """
    if mag.shape != phase.shape:
        raise ValueError("mag and phase must have the same shape.")

    if mag.dtype != torch.float32:
        mag = mag.to(torch.float32)
    if phase.dtype != torch.float32:
        phase = phase.to(torch.float32)

    window = torch.hann_window(win_length, periodic=True, dtype=torch.float32, device=mag.device)

    spec = mag * torch.exp(1j * phase)

    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        length=length,
    )
    return wav


def freq_minmax_normalize(
    mag: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a magnitude spectrogram along the frequency axis to [0,1],
    using per-frequency min/max across time.

    mag: [F, T]
    Returns:
      mag_norm: [F, T]
      f_min: [F, 1]
      f_max: [F, 1]
    """
    if mag.ndim != 2:
        raise ValueError("mag must have shape [F, T].")

    f_min = torch.amin(mag, dim=1, keepdim=True)
    f_max = torch.amax(mag, dim=1, keepdim=True)
    denom = torch.clamp(f_max - f_min, min=eps)
    mag_norm = (mag - f_min) / denom
    mag_norm = torch.clamp(mag_norm, 0.0, 1.0)
    return mag_norm, f_min, f_max


def freq_minmax_denormalize(
    mag_norm: torch.Tensor,
    f_min: torch.Tensor,
    f_max: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Inverse of freq_minmax_normalize.

    mag_norm: [F, T]
    f_min/f_max: [F, 1]
    """
    if mag_norm.ndim != 2:
        raise ValueError("mag_norm must have shape [F, T].")
    if f_min.ndim != 2 or f_max.ndim != 2:
        raise ValueError("f_min and f_max must have shape [F, 1].")
    if f_min.shape[0] != mag_norm.shape[0] or f_max.shape[0] != mag_norm.shape[0]:
        raise ValueError("f_min/f_max must match mag_norm frequency dimension.")

    denom = torch.clamp(f_max - f_min, min=eps)
    mag = mag_norm * denom + f_min
    return mag
