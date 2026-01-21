from pathlib import Path
from typing import Dict, Tuple, Union

import torch

from src.models.unet_2d import UNet2D
from src.training.stft import freq_minmax_normalize


class InferenceConfig:
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: int = 4096,
        center: bool = False,
        eps: float = 1e-8,
        ola_denom_floor: float = 1e-3,
    ):
        """Inference settings for STFT and overlap-add iSTFT."""
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")
        if not isinstance(n_fft, int) or n_fft <= 0:
            raise ValueError("n_fft must be a positive int")
        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError("hop_length must be a positive int")
        if not isinstance(win_length, int) or win_length <= 0:
            raise ValueError("win_length must be a positive int")
        if win_length > n_fft:
            raise ValueError("win_length must be <= n_fft")
        if hop_length > win_length:
            raise ValueError("hop_length must be <= win_length")
        if not isinstance(center, bool):
            raise ValueError("center must be a bool")
        if not isinstance(eps, float) or eps <= 0.0:
            raise ValueError("eps must be a positive float")
        if not isinstance(ola_denom_floor, float) or ola_denom_floor <= 0.0:
            raise ValueError("ola_denom_floor must be a positive float")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.eps = eps
        self.ola_denom_floor = ola_denom_floor


def load_torchscript_model(
    checkpoint_path: Union[str, Path], device: str = "cpu"
) -> torch.jit.ScriptModule:
    """Load a TorchScript model (.pt)."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = torch.device(device)
    model = torch.jit.load(str(path), map_location=dev)
    model.eval()
    return model


def load_pth_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    stems: int = 4,
) -> Tuple[torch.nn.Module, dict]:
    """Load a training checkpoint dict (.pth) containing 'model_state'."""
    if not isinstance(stems, int) or stems <= 0:
        raise ValueError("stems must be a positive int")

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a .pth file, got a directory: {path}")

    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint must be a dict.")
    if "model_state" not in ckpt:
        raise RuntimeError("Checkpoint dict missing 'model_state'.")

    dev = torch.device(device)
    model = UNet2D(stems=stems).to(dev)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    model.eval()
    return model, ckpt


def config_from_checkpoint(ckpt: dict) -> InferenceConfig:
    """Extract inference config from checkpoint extra['config'] if present."""
    if not isinstance(ckpt, dict):
        raise ValueError("ckpt must be a dict")

    cfg = {}
    extra = ckpt.get("extra")
    if isinstance(extra, dict):
        maybe = extra.get("config")
        if isinstance(maybe, dict):
            cfg = maybe

    sample_rate = int(cfg.get("sample_rate", 44100))
    n_fft = int(cfg.get("n_fft", 4096))
    hop_length = int(cfg.get("hop_length", 1024))
    win_length = int(cfg.get("win_length", 4096))
    center = bool(cfg.get("center", False))

    return InferenceConfig(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        eps=1e-8,
        ola_denom_floor=1e-3,
    )


def _istft_overlap_add_no_check(
    X: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    length: int,
    eps: float,
    denom_floor: float,
) -> torch.Tensor:
    """Manual overlap-add iSTFT with a safe denominator floor."""
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch.Tensor")
    if X.ndim != 2:
        raise ValueError("X must have shape (F, T)")
    if not torch.is_complex(X):
        raise ValueError("X must be a complex tensor")

    if not isinstance(n_fft, int) or n_fft <= 0:
        raise ValueError("n_fft must be a positive int")
    if not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError("hop_length must be a positive int")
    if not isinstance(win_length, int) or win_length <= 0:
        raise ValueError("win_length must be a positive int")
    if win_length > n_fft:
        raise ValueError("win_length must be <= n_fft")
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive int")
    if not isinstance(eps, float) or eps <= 0.0:
        raise ValueError("eps must be a positive float")
    if not isinstance(denom_floor, float) or denom_floor <= 0.0:
        raise ValueError("denom_floor must be a positive float")

    if not isinstance(window, torch.Tensor):
        raise ValueError("window must be a torch.Tensor")
    if window.ndim != 1 or int(window.numel()) != win_length:
        raise ValueError("window must have shape (win_length,)")

    F, T = int(X.shape[0]), int(X.shape[1])
    expected_F = n_fft // 2 + 1
    if F != expected_F:
        raise ValueError(f"X first dimension must be {expected_F} for n_fft={n_fft}, got {F}")

    frames = torch.fft.irfft(X, n=n_fft, dim=0)  # (n_fft, T)
    frames = frames[:win_length, :]  # (win_length, T)
    frames = frames * window.unsqueeze(1)  # (win_length, T)

    out_len = win_length + hop_length * (T - 1)
    y = torch.zeros(out_len, device=frames.device, dtype=frames.dtype)
    denom = torch.zeros(out_len, device=frames.device, dtype=frames.dtype)

    w_sq = (window * window).to(dtype=frames.dtype)

    for t in range(T):
        start = t * hop_length
        end = start + win_length
        y[start:end] = y[start:end] + frames[:, t]
        denom[start:end] = denom[start:end] + w_sq

    floor = max(float(eps), float(denom_floor))
    denom = torch.clamp(denom, min=floor)
    y = y / denom

    if out_len >= length:
        return y[:length]
    pad = torch.zeros(length - out_len, device=y.device, dtype=y.dtype)
    return torch.cat([y, pad], dim=0)


def separate_waveform_4stems(
    waveform: torch.Tensor,
    cfg: InferenceConfig,
    model,
    device: str = "cpu",
    renorm_sum_to_one: bool = False,
) -> Dict[str, torch.Tensor]:
    """Separate waveform into 4 stems using ratio masks."""
    if not isinstance(cfg, InferenceConfig):
        raise ValueError("cfg must be an InferenceConfig")
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("waveform must be a torch.Tensor")
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape (channels, samples)")
    if not callable(model):
        raise ValueError("model must be callable")
    if not isinstance(renorm_sum_to_one, bool):
        raise ValueError("renorm_sum_to_one must be a bool")

    channels = int(waveform.shape[0])
    n_samples = int(waveform.shape[1])
    if channels <= 0 or n_samples <= 0:
        raise ValueError("waveform must have non-empty shape (channels, samples)")

    dev = torch.device(device)
    x = waveform.to(device=dev, dtype=torch.float32)

    window = torch.hann_window(cfg.win_length, periodic=True, dtype=torch.float32, device=dev)

    stfts = []
    for ch in range(channels):
        stft_c = torch.stft(
            x[ch],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=cfg.center,
            return_complex=True,
        )
        stfts.append(stft_c)
    stft = torch.stack(stfts, dim=0)  # (C, F, T)

    x_mono = x.mean(dim=0)
    stft_mono = torch.stft(
        x_mono,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        return_complex=True,
    )  # (F, T)
    mix_mag_mono = torch.abs(stft_mono)

    mix_norm, _, _ = freq_minmax_normalize(mix_mag_mono, eps=cfg.eps)
    model_in = mix_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)

    with torch.no_grad():
        pred = model(model_in)

    if not isinstance(pred, torch.Tensor) or pred.ndim != 4:
        raise ValueError("model must return a Tensor of shape (1,4,F,T)")
    if pred.shape[0] != 1 or pred.shape[1] != 4:
        raise ValueError("model must return a Tensor of shape (1,4,F,T)")
    if pred.shape[2] != stft_mono.shape[0] or pred.shape[3] != stft_mono.shape[1]:
        raise ValueError("model output shape must match mono STFT (F,T)")
    if stft.shape[1] != stft_mono.shape[0] or stft.shape[2] != stft_mono.shape[1]:
        raise ValueError("per-channel STFT shape must match mono STFT shape")

    masks = torch.clamp(pred[0], 0.0, 1.0)  # (4, F, T)

    if renorm_sum_to_one:
        s = masks.sum(dim=0)  # (F, T)
        denom = torch.clamp(s, min=cfg.eps)
        masks = masks / denom.unsqueeze(0)

    names = ["drums", "bass", "vocals", "other"]
    stems: Dict[str, torch.Tensor] = {}

    mix_phase = torch.angle(stft)  # (C, F, T)
    mix_mag = torch.abs(stft)  # (C, F, T)

    for i, name in enumerate(names):
        outs = []
        for ch in range(channels):
            stem_mag = masks[i] * mix_mag[ch]
            spec = torch.polar(stem_mag, mix_phase[ch])

            y = _istft_overlap_add_no_check(
                spec,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                window=window,
                length=n_samples,
                eps=cfg.eps,
                denom_floor=cfg.ola_denom_floor,
            )

            if int(y.numel()) > n_samples:
                y = y[:n_samples]
            elif int(y.numel()) < n_samples:
                y = torch.nn.functional.pad(y, (0, n_samples - int(y.numel())))

            outs.append(y.detach().cpu())
        stems[name] = torch.stack(outs, dim=0)

    return stems
