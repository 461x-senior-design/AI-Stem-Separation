import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from src.constants import STEMS_4
from src.inference import InferenceConfig, config_from_checkpoint, load_pth_model, separate_audio_file

STEM_NAMES: list[str] = STEMS_4


def validate_file(path: Path, description: str) -> None:
    """Raise if `path` does not exist or is not a file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file for {description}: {path}")


def validate_dir(path: Path, description: str) -> None:
    """Raise if `path` does not exist or is not a directory."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory for {description}: {path}")


def get_env_path(name: str) -> Optional[Path]:
    """Return an environment variable as an absolute Path, or None if unset/blank."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable, with a safe default."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from e


def resolve_checkpoint() -> Path:
    """Resolve the checkpoint path from CKPT or a repo default."""
    ckpt = get_env_path("CKPT")
    if ckpt is not None:
        validate_file(ckpt, "checkpoint (.pth)")
        return ckpt

    default_ckpt = Path("runs/best_ckpt/unet_phase1_best.pth").expanduser().resolve()
    if default_ckpt.exists():
        validate_file(default_ckpt, "checkpoint (.pth)")
        return default_ckpt

    raise RuntimeError(
        "Set CKPT to a .pth checkpoint path, i.e. runs/best_ckpt/unet_phase1_best.pth"
    )


def resolve_track_dir() -> Path:
    """Resolve the MUSDB track directory from TRACK_DIR."""
    track_dir = get_env_path("TRACK_DIR")
    if track_dir is None:
        raise RuntimeError(
            "Set TRACK_DIR to a MUSDB track directory (contains mixture.wav and stem wavs)."
        )
    validate_dir(track_dir, "TRACK_DIR (MUSDB track directory)")
    return track_dir


def read_mixture_stereo(mix_path: Path, max_seconds: int) -> tuple[np.ndarray, int]:
    """Read mixture.wav as float32 stereo (N,2), optionally truncated to max_seconds."""
    validate_file(mix_path, "mixture.wav")

    info = sf.info(str(mix_path))
    sr = int(info.samplerate)

    if max_seconds > 0:
        frames = int(sr * max_seconds)
        mix, read_sr = sf.read(
            str(mix_path),
            frames=frames,
            always_2d=True,
            dtype="float32",
        )
    else:
        mix, read_sr = sf.read(
            str(mix_path),
            always_2d=True,
            dtype="float32",
        )

    if int(read_sr) != sr:
        raise RuntimeError(f"soundfile returned unexpected sample rate: {read_sr} != {sr}")

    if mix.ndim != 2:
        raise RuntimeError(f"Expected mixture to be 2D (N,C), got shape {mix.shape}")

    if mix.shape[1] != 2:
        raise RuntimeError(f"Expected stereo mixture.wav with shape (N,2), got {mix.shape}")

    return mix, sr


def main() -> None:
    """Run stem separation on one MUSDB track and write stems to OUT_DIR."""
    ckpt_path = resolve_checkpoint()
    track_dir = resolve_track_dir()

    out_dir = Path(os.environ.get("OUT_DIR", "runs/separated_demo")).expanduser().resolve()
    max_seconds = get_env_int("MAX_SECONDS", 60)

    mix_path = track_dir / "mixture.wav"
    mix, sr = read_mixture_stereo(mix_path, max_seconds=max_seconds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_obj = load_pth_model(str(ckpt_path), device=device, stems=len(STEM_NAMES))
    cfg = config_from_checkpoint(ckpt_obj)
    cfg = replace(cfg, device=device, renorm_masks=True)
    model.eval()

    if sr != cfg.sample_rate:
        raise RuntimeError(
            f"Sample-rate mismatch: mixture is {sr}, config expects {cfg.sample_rate}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = separate_audio_file(
        audio_path=mix_path,
        model=model,
        cfg=cfg,
        stem_names=STEM_NAMES,
        output_dir=out_dir,
        export_files=True,
    )

    pred_np: dict[str, np.ndarray] = {}
    for name, waveform in outputs.waveforms.items():
        pred_np[name] = waveform.T

    # Diagnostics
    # energy check per stem
    for name in STEM_NAMES:
        x = pred_np[name]
        rms = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        print(f"{name:6s} rms={rms:.6f} peak={peak:.6f}")

    # edge spike check
    edge = int(cfg.win_length)
    if mix.shape[0] > 2 * edge:
        for name in STEM_NAMES:
            x = pred_np[name]
            inner = x[edge:-edge, :]
            inner_peak = float(np.max(np.abs(inner)))
            print(f"{name:6s} inner_peak={inner_peak:.6f}")

    # overall check
    mix_rms = float(np.sqrt(np.mean(mix * mix)))
    mix_peak = float(np.max(np.abs(mix)))
    print(f"mix    rms={mix_rms:.6f} peak={mix_peak:.6f}")

    stems_sum = pred_np["drums"] + pred_np["bass"] + pred_np["vocals"] + pred_np["other"]
    err = mix - stems_sum
    mix_pow = float(np.mean(mix * mix) + 1e-12)
    err_pow = float(np.mean(err * err) + 1e-12)
    snr = 10.0 * np.log10(mix_pow / err_pow)

    seconds_used = float(mix.shape[0]) / float(sr)
    print("TRACK_DIR:", str(track_dir))
    print("CKPT:", str(ckpt_path))
    print("OUT_DIR:", str(out_dir))
    print("device:", device)
    print("seconds_used:", seconds_used)
    print("recon_snr_db:", snr)


if __name__ == "__main__":
    main()
