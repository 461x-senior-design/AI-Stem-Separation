# src/stemmy/training/losses.py
"""Auxiliary loss functions for spectrogram-mask U-Net training.

- multi_scale_spec_loss: multi-scale spectrogram L1 loss (Exp C)
- si_sdr_loss: negative mean SI-SDR on waveforms (Exp D helper)
- compute_sisdr_loss_from_masks: SI-SDR loss via ISTFT reconstruction (Exp D)
"""

import torch
import torch.nn.functional as F


def multi_scale_spec_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mix_mag_unnorm: torch.Tensor,
    scales: tuple = (1, 2, 4),
) -> torch.Tensor:
    """Multi-scale spectrogram L1 loss.

    For each temporal scale, computes L1 between predicted stem magnitudes
    and target stem magnitudes. Predicted magnitudes are softmax(logits) * mix_mag.

    Args:
        logits: [B, S, F, T] raw model output
        targets: [B, S, F, T] ratio masks (sum-to-one across S)
        mix_mag_unnorm: [B, 1, F, T] unnormalized mixture magnitude
        scales: temporal pooling factors (1 = full resolution)

    Returns:
        Scalar loss (mean over scales).
    """
    B, S, F_dim, T_dim = logits.shape
    pred_masks = torch.softmax(logits, dim=1)
    pred_mags = pred_masks * mix_mag_unnorm   # [B, S, F, T] via broadcast
    target_mags = targets * mix_mag_unnorm    # [B, S, F, T] via broadcast

    total = logits.new_zeros(())
    for scale in scales:
        if scale > 1:
            # avg_pool2d expects [N, C, H, W]; use [B*S, 1, F, T]
            p = F.avg_pool2d(
                pred_mags.reshape(B * S, 1, F_dim, T_dim), (1, scale)
            ).reshape(B, S, F_dim, T_dim // scale)
            t = F.avg_pool2d(
                target_mags.reshape(B * S, 1, F_dim, T_dim), (1, scale)
            ).reshape(B, S, F_dim, T_dim // scale)
        else:
            p, t = pred_mags, target_mags
        total = total + F.l1_loss(p, t)

    return total / len(scales)


def si_sdr_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative mean SI-SDR loss over a batch of stems.

    Args:
        pred_wav: [B, S, N] predicted waveforms
        target_wav: [B, S, N] target waveforms
        eps: numerical stability

    Returns:
        Scalar loss (negative mean SI-SDR; minimize to improve SI-SDR).
    """
    target_energy = (target_wav ** 2).sum(-1, keepdim=True).clamp(min=eps)
    dot = (pred_wav * target_wav).sum(-1, keepdim=True)
    scale = dot / target_energy
    projection = scale * target_wav
    noise = pred_wav - projection
    signal_power = (projection ** 2).sum(-1)
    noise_power = (noise ** 2).sum(-1).clamp(min=eps)
    ratio = signal_power / noise_power
    sisdr = 10.0 * torch.log10(ratio.clamp(min=eps))
    return -sisdr.mean()


def compute_sisdr_loss_from_masks(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mix_mag_unnorm: torch.Tensor,
    mix_phase: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SI-SDR loss via ISTFT reconstruction of predicted and target stems.

    Reconstructs each stem waveform by applying softmax masks (predicted) or
    ratio masks (target) to the complex mixture STFT, then running ISTFT.

    Note: ISTFT always uses center=True internally because the Hann window has
    value 0 at position 0, making the overlap-add sum 0 at the boundary when
    center=False (which causes PyTorch to raise an error). Using center=True for
    both pred and target waveforms produces a consistent reconstruction that is
    valid as a differentiable training signal even when the forward STFT used
    center=False.

    Args:
        logits: [B, S, F, T] raw model output (float32)
        targets: [B, S, F, T] ratio masks (float32)
        mix_mag_unnorm: [B, 1, F, T] unnormalized mixture magnitude (float32)
        mix_phase: [B, 1, F, T] mixture phase in radians (float32)
        n_fft: FFT size
        hop_length: hop length
        win_length: window length
        window: [win_length] window tensor on correct device
        eps: numerical stability

    Returns:
        Scalar SI-SDR loss.
    """
    B, S, F_dim, T_dim = logits.shape

    # All in float32 for ISTFT stability
    logits_f = logits.float()
    targets_f = targets.float()
    mag_f = mix_mag_unnorm.float()[:, 0]   # [B, F, T]
    phase_f = mix_phase.float()[:, 0]      # [B, F, T]

    pred_masks = torch.softmax(logits_f, dim=1)  # [B, S, F, T]

    # Complex mix STFT: [B, F, T]
    mix_complex = torch.polar(mag_f, phase_f)

    # [B, S, F, T] complex
    pred_complex = pred_masks * mix_complex.unsqueeze(1)
    tgt_complex = targets_f * mix_complex.unsqueeze(1)

    # Flatten to [B*S, F, T] for batched ISTFT
    pred_flat = pred_complex.reshape(B * S, F_dim, T_dim)
    tgt_flat = tgt_complex.reshape(B * S, F_dim, T_dim)

    # center=True: Hann window OLA is valid everywhere (avoids OLA=0 at boundary
    # which occurs with center=False). Both pred and target use the same center
    # setting so SI-SDR between them is unaffected by this choice.
    window_f = window.float()
    pred_wav_flat = torch.istft(
        pred_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_f,
        center=True,
    )
    tgt_wav_flat = torch.istft(
        tgt_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_f,
        center=True,
    )

    # Reshape: [B, S, N]
    N = pred_wav_flat.shape[-1]
    pred_wav = pred_wav_flat.reshape(B, S, N)
    tgt_wav = tgt_wav_flat.reshape(B, S, N)

    return si_sdr_loss(pred_wav, tgt_wav, eps=eps)
