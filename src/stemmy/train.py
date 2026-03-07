# src/stemmy/train.py
"""Training entry point for the spectrogram-mask U-Net on MUSDB18-HQ.

This script trains a UNet2D to predict per-stem ratio masks from a normalized mono
mixture magnitude spectrogram.

Key invariants:
- STFT configuration is centralized in stemmy.constants.
- The STFT `center` setting (stemmy.constants.STFT_CENTER) must match across
  preprocessing, training, inference, and reconstruction.
- Dataset crops are chosen so each example produces exactly T time frames.

Outputs:
- Checkpoints (*.pth) written to --checkpoint-dir, containing model+optimizer state
  and a small config dict for reproducibility.
- Optional TorchScript export (*.pt) via --export-ts.

Loss composition (all terms off by default; enable via CLI flags):
- KL divergence (always active): KL(target || softmax(logits))
- STFT loss (--stft-loss-weight > 0): multi-scale spectrogram L1 (Exp C)
- SI-SDR loss (--sisdr-loss-weight > 0): negative SI-SDR via ISTFT (Exp D)
"""

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

import wandb
from stemmy.constants import (
    BAR_FINISHED_STYLE,
    BOLD_PURPLE,
    BOLD_VIOLET,
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    LAVENDER,
    LOSS_STYLE,
    N_FFT,
    NEON_GREEN,
    ROSE_RED,
    STEMS_4,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WHITE,
    WIN_LENGTH,
    WINDOW,
)
from stemmy.logging_config import get_logger, setup_logging
from stemmy.models.unet_2d import UNet2D
from stemmy.tool.progress_theme import (
    create_setup_progress,
    create_themed_progress,
    start_eq_animator,
)
from stemmy.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from stemmy.training.losses import compute_sisdr_loss_from_masks, multi_scale_spec_loss
from stemmy.training.musdb18hq_dataset import CropConfig, GainAugConfig, Musdb18HQDataset
from stemmy.training.stft import StftConfig
from stemmy.wandb_config import wandb_run

logger = get_logger(__name__)

_LOSS_EPS: float = 1e-8


def _progress_disabled() -> bool:
    """Return True when terminal progress rendering should be disabled."""
    return os.getenv("STEMMY_DISABLE_PROGRESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train U-Net on MUSDB18-HQ (baseline).")

    p.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Root directory containing the dataset splits used by the dataset class.",
    )

    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW (>= 0).",
    )

    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")

    p.add_argument(
        "--waveform-norm",
        type=str,
        default=DEFAULT_WAVEFORM_NORM,
        choices=["peak", "rms", "none"],
        help="Waveform normalization method (applies to mix and stems).",
    )

    p.add_argument(
        "--spectrogram-norm",
        type=str,
        default=DEFAULT_SPECTROGRAM_NORM,
        choices=["freq_minmax", "none"],
        help="Spectrogram normalization method for model inputs.",
    )

    p.add_argument(
        "--time-frames",
        type=int,
        default=256,
        help="Fixed STFT time frames T (256 or 512).",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="If >0, limit number of tracks per split.",
    )

    p.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base channel count for the U-Net (architecture width).",
    )

    p.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor in (0,1).",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="ReduceLROnPlateau patience (epochs).",
    )
    p.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="ReduceLROnPlateau minimum LR (> 0).",
    )

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Output directory.")
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a .pth checkpoint to resume from.",
    )
    p.add_argument(
        "--save-every-epochs",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs.",
    )

    p.add_argument(
        "--export-ts",
        action="store_true",
        help="Export TorchScript .pt after each checkpoint save.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cpu, cuda, or leave empty for auto selection.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="If >= 0, seed RNGs for reproducible training behavior.",
    )

    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP (only applies on CUDA).",
    )

    # --- Exp A: Gradient clipping ---
    p.add_argument(
        "--clip-grad-norm",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping (0.0 = disabled). Recommended: 1.0.",
    )

    # --- Exp B: Gain augmentation ---
    p.add_argument(
        "--gain-aug",
        action="store_true",
        help="Enable per-stem gain augmentation on the training set.",
    )
    p.add_argument(
        "--gain-aug-min",
        type=float,
        default=0.25,
        help="Minimum per-stem gain multiplier for gain augmentation.",
    )
    p.add_argument(
        "--gain-aug-max",
        type=float,
        default=1.75,
        help="Maximum per-stem gain multiplier for gain augmentation.",
    )

    # --- Exp C: Multi-scale spectrogram loss ---
    p.add_argument(
        "--stft-loss-weight",
        type=float,
        default=0.0,
        help="Weight for multi-scale spectrogram L1 loss (0.0 = disabled). Try 0.1.",
    )

    # --- Exp D: SI-SDR loss ---
    p.add_argument(
        "--sisdr-loss-weight",
        type=float,
        default=0.0,
        help="Weight for SI-SDR loss via ISTFT (0.0 = disabled). Try 0.1.",
    )

    # --- W&B experiment tag ---
    p.add_argument(
        "--experiment",
        type=str,
        default="",
        help="Experiment identifier logged to W&B config (e.g. 'clip_only', 'gain_aug').",
    )

    return p.parse_args(argv)


def pick_device(arg: str) -> torch.device:
    """Choose torch.device based on user input and CUDA availability."""
    if arg.strip():
        val = arg.strip()
        if val == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but CUDA is not available.")
        return torch.device(val)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument values early so failures are clear."""
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.time_frames not in [256, 512]:
        raise ValueError("--time-frames must be 256 or 512 for this baseline.")
    if args.save_every_epochs <= 0:
        raise ValueError("--save-every-epochs must be > 0")
    if args.max_tracks < 0:
        raise ValueError("--max-tracks must be >= 0")
    if args.base_channels <= 0:
        raise ValueError("--base-channels must be > 0")

    if args.lr_factor <= 0.0 or args.lr_factor >= 1.0:
        raise ValueError("--lr-factor must be in (0, 1)")
    if args.lr_patience < 0:
        raise ValueError("--lr-patience must be >= 0")
    if args.min_lr <= 0:
        raise ValueError("--min-lr must be > 0")

    if not isinstance(args.data_root, str) or args.data_root.strip() == "":
        raise ValueError("--data-root must be a non-empty string")
    if not isinstance(args.checkpoint_dir, str) or args.checkpoint_dir.strip() == "":
        raise ValueError("--checkpoint-dir must be a non-empty string")

    data_root_p = Path(args.data_root.strip()).expanduser()
    if not data_root_p.exists():
        raise FileNotFoundError(f"--data-root not found: {data_root_p}")
    if not data_root_p.is_dir():
        raise NotADirectoryError(f"--data-root must be a directory: {data_root_p}")

    ckpt_dir_p = Path(args.checkpoint_dir.strip()).expanduser()
    ckpt_dir_p.mkdir(parents=True, exist_ok=True)
    if not ckpt_dir_p.is_dir():
        raise NotADirectoryError(f"--checkpoint-dir must be a directory: {ckpt_dir_p}")

    if args.resume.strip():
        resume_p = Path(args.resume.strip()).expanduser()
        if not resume_p.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_p}")
        if resume_p.is_dir():
            raise IsADirectoryError(
                f"--resume must point to a .pth file, got directory: {resume_p}"
            )

    if not isinstance(args.seed, int):
        raise ValueError("--seed must be an int")
    if args.seed < -1:
        raise ValueError("--seed must be -1 (disabled) or >= 0")

    if args.clip_grad_norm < 0.0:
        raise ValueError("--clip-grad-norm must be >= 0.0")

    if args.gain_aug_min <= 0.0:
        raise ValueError("--gain-aug-min must be > 0")
    if args.gain_aug_max <= args.gain_aug_min:
        raise ValueError("--gain-aug-max must be > --gain-aug-min")

    if args.stft_loss_weight < 0.0:
        raise ValueError("--stft-loss-weight must be >= 0.0")
    if args.sisdr_loss_weight < 0.0:
        raise ValueError("--sisdr-loss-weight must be >= 0.0")


def set_seed(seed: int) -> None:
    """Seed RNGs for reproducible behavior (without forcing full determinism)."""
    if seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    args: argparse.Namespace,
    stft_cfg: StftConfig,
    crop_cfg: CropConfig,
    device: torch.device,
) -> tuple[Musdb18HQDataset, Musdb18HQDataset, DataLoader, DataLoader]:
    """Build train/val datasets and DataLoaders."""
    max_tracks = args.max_tracks if args.max_tracks > 0 else None

    train_gain_cfg = GainAugConfig(
        enabled=bool(args.gain_aug),
        min_gain=args.gain_aug_min,
        max_gain=args.gain_aug_max,
    )
    val_gain_cfg = GainAugConfig(enabled=False)

    train_ds = Musdb18HQDataset(
        root_dir=args.data_root,
        split="train",
        stft_cfg=stft_cfg,
        crop_cfg=crop_cfg,
        stems=STEMS_4,
        max_tracks=max_tracks,
        deterministic=False,
        waveform_norm=args.waveform_norm,
        spectrogram_norm=args.spectrogram_norm,
        seed=0,
        gain_aug_cfg=train_gain_cfg,
    )
    val_ds = Musdb18HQDataset(
        root_dir=args.data_root,
        split="test",
        stft_cfg=stft_cfg,
        crop_cfg=crop_cfg,
        stems=STEMS_4,
        max_tracks=max_tracks,
        deterministic=True,
        waveform_norm=args.waveform_norm,
        spectrogram_norm=args.spectrogram_norm,
        seed=0,
        gain_aug_cfg=val_gain_cfg,
    )

    train_drop_last = len(train_ds) >= args.batch_size
    if not train_drop_last:
        logger.warning(
            "Train dataset size (%d) is smaller than batch_size (%d); disabling drop_last "
            "to avoid zero training batches.",
            int(len(train_ds)),
            int(args.batch_size),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=train_drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_run_config(
    args: argparse.Namespace,
    device: torch.device,
    stft_cfg: StftConfig,
    crop_cfg: CropConfig,
) -> dict:
    """Build a small config dict stored inside checkpoints for reproducibility."""
    return {
        "experiment": args.experiment,
        "data_root": args.data_root,
        "sample_rate": stft_cfg.sample_rate,
        "n_fft": stft_cfg.n_fft,
        "hop_length": stft_cfg.hop_length,
        "win_length": stft_cfg.win_length,
        "center": bool(stft_cfg.center),
        "window": stft_cfg.window,
        "time_frames": crop_cfg.time_frames,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "device": str(device),
        "stems": STEMS_4,
        "max_tracks": args.max_tracks,
        "waveform_norm": args.waveform_norm,
        "spectrogram_norm": args.spectrogram_norm,
        "loss": "kl_div(target||pred)",
        "pred_activation": "softmax_over_stems",
        "target_mask_mode": "stem_mag / sum_stem_mags",
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "clip_grad_norm": args.clip_grad_norm,
        "gain_aug": bool(args.gain_aug),
        "gain_aug_min": args.gain_aug_min,
        "gain_aug_max": args.gain_aug_max,
        "stft_loss_weight": args.stft_loss_weight,
        "sisdr_loss_weight": args.sisdr_loss_weight,
    }


def kl_div_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """KL divergence KL(target || pred) with pred derived from logits via softmax.

    Args:
        logits: [B, S, F, T]
        target: [B, S, F, T] distribution over stems (sum-to-one across S)
        eps: small positive float for numerical stability

    Returns:
        Scalar loss (mean over B,F,T; sum over S).
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits and target must have same shape, got {logits.shape} vs {target.shape}"
        )

    log_pred = F.log_softmax(logits, dim=1)
    target_safe = torch.clamp(target, min=float(eps))

    loss_map = target_safe * (torch.log(target_safe) - log_pred)
    loss = torch.sum(loss_map, dim=1).mean()
    return loss


def _make_grad_scaler(device: torch.device, use_amp: bool) -> Any:
    """Create an AMP GradScaler without triggering deprecation warnings on new PyTorch."""
    if not use_amp:
        try:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                return torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=False)
        return torch.cuda.amp.GradScaler(enabled=False)

    if device.type != "cuda":
        return torch.cuda.amp.GradScaler(enabled=False)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)

    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(device: torch.device, use_amp: bool):
    """Return an autocast context manager without triggering deprecation warnings."""
    if not use_amp or device.type != "cuda":
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except TypeError:
            try:
                return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            except TypeError:
                return torch.cuda.amp.autocast(dtype=torch.float16)

    return torch.cuda.amp.autocast(dtype=torch.float16)


def print_training_summary(rows: list[dict[str, Any]], include_torchscript: bool = False) -> None:
    """Print an end-of-run training summary table."""
    console = Console()
    table = Table(title="Training Summary", header_style=BOLD_PURPLE)
    table.add_column("Epoch", justify="right", style=WHITE)
    table.add_column("Train Loss", justify="right", style=WHITE)
    table.add_column("Val Loss", justify="right", style=WHITE)
    table.add_column("Checkpoint", style=LAVENDER)
    if include_torchscript:
        table.add_column("TorchScript", style=LAVENDER)

    best_train = min((row["train_loss"] for row in rows), default=None)
    worst_train = max((row["train_loss"] for row in rows), default=None)
    best_val = min((row["val_loss"] for row in rows), default=None)
    worst_val = max((row["val_loss"] for row in rows), default=None)

    def style_loss(value: float, best: Optional[float], worst: Optional[float]) -> str:
        if best is None or worst is None:
            return f"[{WHITE}]{value:.6f}[/]"
        if value == best:
            return f"[{NEON_GREEN}]{value:.6f}[/]"
        if value == worst:
            return f"[{ROSE_RED}]{value:.6f}[/]"
        return f"[{WHITE}]{value:.6f}[/]"

    def display_path(path_value: str) -> str:
        if path_value == "-":
            return path_value
        return Path(path_value).name

    for row in rows:
        cells = [
            str(row["epoch"]),
            style_loss(float(row["train_loss"]), best_train, worst_train),
            style_loss(float(row["val_loss"]), best_val, worst_val),
            display_path(str(row["checkpoint_path"])),
        ]
        if include_torchscript:
            cells.append(display_path(str(row["torchscript_path"])))
        table.add_row(*cells)

    console.print()
    console.print(table)


def _run_batch(
    model: torch.nn.Module,
    mix_norm: torch.Tensor,
    targets_norm: torch.Tensor,
    mix_mag_unnorm: torch.Tensor,
    mix_phase: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Any,
    clip_grad_norm: float,
    stft_loss_weight: float,
    sisdr_loss_weight: float,
    stft_cfg: Optional[StftConfig],
    istft_window: Optional[torch.Tensor],
) -> tuple[float, float, float, float]:
    """Run a single training batch and return (kl_loss, stft_loss, sisdr_loss, total_loss)."""
    optimizer.zero_grad(set_to_none=True)

    if use_amp:
        with _autocast_context(device, use_amp):
            logits = model(mix_norm)
            kl_loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)
            if stft_loss_weight > 0.0:
                stft_l = multi_scale_spec_loss(logits, targets_norm, mix_mag_unnorm)
            else:
                stft_l = torch.zeros((), device=device)

        if sisdr_loss_weight > 0.0 and istft_window is not None and stft_cfg is not None:
            sisdr_l = compute_sisdr_loss_from_masks(
                logits.float(),
                targets_norm.float(),
                mix_mag_unnorm.float(),
                mix_phase.float(),
                n_fft=stft_cfg.n_fft,
                hop_length=stft_cfg.hop_length,
                win_length=stft_cfg.win_length,
                window=istft_window,
            )
        else:
            sisdr_l = torch.zeros((), device=device)

        total_loss = kl_loss + stft_loss_weight * stft_l + sisdr_loss_weight * sisdr_l
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(mix_norm)
        kl_loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)

        if stft_loss_weight > 0.0:
            stft_l = multi_scale_spec_loss(logits, targets_norm, mix_mag_unnorm)
        else:
            stft_l = torch.zeros((), device=device)

        if sisdr_loss_weight > 0.0 and istft_window is not None and stft_cfg is not None:
            sisdr_l = compute_sisdr_loss_from_masks(
                logits,
                targets_norm,
                mix_mag_unnorm,
                mix_phase,
                n_fft=stft_cfg.n_fft,
                hop_length=stft_cfg.hop_length,
                win_length=stft_cfg.win_length,
                window=istft_window,
            )
        else:
            sisdr_l = torch.zeros((), device=device)

        total_loss = kl_loss + stft_loss_weight * stft_l + sisdr_loss_weight * sisdr_l
        total_loss.backward()
        if clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

    return (
        float(kl_loss.detach().cpu().item()),
        float(stft_l.detach().cpu().item()),
        float(sisdr_l.detach().cpu().item()),
        float(total_loss.detach().cpu().item()),
    )


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    scaler: Any,
    epoch_index: int,
    clip_grad_norm: float = 0.0,
    stft_loss_weight: float = 0.0,
    sisdr_loss_weight: float = 0.0,
    stft_cfg: Optional[StftConfig] = None,
) -> tuple[dict[str, float], int]:
    """Run one training epoch and return (loss_dict, num_batches).

    loss_dict keys: 'kl', 'stft', 'sisdr', 'total'
    """
    model.train()

    # Build ISTFT window once if SI-SDR loss is active
    istft_window: Optional[torch.Tensor] = None
    if sisdr_loss_weight > 0.0 and stft_cfg is not None:
        istft_window = torch.hann_window(stft_cfg.win_length, device=device)

    running_kl = 0.0
    running_stft = 0.0
    running_sisdr = 0.0
    running_total = 0.0
    batches = 0

    if _progress_disabled():
        logger.info("Epoch %d train: %d batches", int(epoch_index + 1), int(len(train_loader)))
        for mix_norm, targets_norm, mix_mag_unnorm, mix_phase in train_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)
            mix_mag_unnorm = mix_mag_unnorm.to(device, non_blocking=True)
            mix_phase = mix_phase.to(device, non_blocking=True)

            kl, stft, sisdr, total = _run_batch(
                model, mix_norm, targets_norm, mix_mag_unnorm, mix_phase,
                optimizer, device, use_amp, scaler,
                clip_grad_norm, stft_loss_weight, sisdr_loss_weight,
                stft_cfg, istft_window,
            )
            running_kl += kl
            running_stft += stft
            running_sisdr += sisdr
            running_total += total
            batches += 1

        n = max(1, batches)
        return {"kl": running_kl / n, "stft": running_stft / n,
                "sisdr": running_sisdr / n, "total": running_total / n}, batches

    with create_themed_progress(
        "Epoch {task.fields[epoch]} Train", title_style=BOLD_PURPLE
    ) as progress:
        train_task = progress.add_task(
            "train",
            total=len(train_loader),
            epoch=epoch_index + 1,
            eq="",
        )
        train_anim_stop, train_anim_thread = start_eq_animator(progress, train_task, fps=8)
        try:
            for mix_norm, targets_norm, mix_mag_unnorm, mix_phase in train_loader:
                mix_norm = mix_norm.to(device, non_blocking=True)
                targets_norm = targets_norm.to(device, non_blocking=True)
                mix_mag_unnorm = mix_mag_unnorm.to(device, non_blocking=True)
                mix_phase = mix_phase.to(device, non_blocking=True)

                kl, stft, sisdr, total = _run_batch(
                    model, mix_norm, targets_norm, mix_mag_unnorm, mix_phase,
                    optimizer, device, use_amp, scaler,
                    clip_grad_norm, stft_loss_weight, sisdr_loss_weight,
                    stft_cfg, istft_window,
                )
                running_kl += kl
                running_stft += stft
                running_sisdr += sisdr
                running_total += total
                batches += 1
                progress.update(train_task, advance=1)
        finally:
            train_anim_stop.set()
            train_anim_thread.join()
            progress.update(train_task, eq="▁▁▁▁▁")

    n = max(1, batches)
    return {"kl": running_kl / n, "stft": running_stft / n,
            "sisdr": running_sisdr / n, "total": running_total / n}, batches


def eval_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch_index: int,
) -> tuple[float, int]:
    """Run validation and return (mean_kl_loss, num_batches)."""
    model.eval()
    running = 0.0
    batches = 0

    if _progress_disabled():
        logger.info("Epoch %d val: %d batches", int(epoch_index + 1), int(len(val_loader)))
        with torch.no_grad():
            for mix_norm, targets_norm, _mix_mag_unnorm, _mix_phase in val_loader:
                mix_norm = mix_norm.to(device, non_blocking=True)
                targets_norm = targets_norm.to(device, non_blocking=True)

                logits = model(mix_norm)
                v_loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)

                running += float(v_loss.cpu().item())
                batches += 1
        mean_loss = running / max(1, batches)
        return mean_loss, batches

    with torch.no_grad():
        with create_themed_progress(
            "Epoch {task.fields[epoch]} Val  ", title_style=BOLD_VIOLET
        ) as progress:
            val_task = progress.add_task(
                "val",
                total=len(val_loader),
                epoch=epoch_index + 1,
                eq="",
            )
            val_anim_stop, val_anim_thread = start_eq_animator(progress, val_task, fps=8)
            try:
                for mix_norm, targets_norm, _mix_mag_unnorm, _mix_phase in val_loader:
                    mix_norm = mix_norm.to(device, non_blocking=True)
                    targets_norm = targets_norm.to(device, non_blocking=True)

                    logits = model(mix_norm)
                    v_loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)

                    running += float(v_loss.cpu().item())
                    batches += 1
                    progress.update(val_task, advance=1)
            finally:
                val_anim_stop.set()
                val_anim_thread.join()
                progress.update(val_task, eq="▁▁▁▁▁")

    mean_loss = running / max(1, batches)
    return mean_loss, batches


@wandb_run(job_type="training", name="training")
def main(argv: Optional[list[str]] = None) -> None:
    """Main training loop."""
    setup_logging()
    progress_disabled = _progress_disabled()

    args = parse_args(argv)
    if progress_disabled:
        logger.info("Progress bars disabled (STEMMY_DISABLE_PROGRESS=1)")

    summary_rows: list[dict[str, Any]] = []
    with create_setup_progress("Setup", title_style=BOLD_PURPLE) as setup_progress:
        setup_task = setup_progress.add_task(
            "Setup", total=11, step="Validating Args...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: validating args")
        validate_args(args)
        setup_progress.update(setup_task, advance=1, step="Seeding RNG...", step_style=LOSS_STYLE)
        if progress_disabled:
            logger.info("Setup: seeding RNG")
        set_seed(args.seed)
        setup_progress.update(
            setup_task, advance=1, step="Picking device...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: picking device")
        device = pick_device(args.device)
        setup_progress.update(
            setup_task, advance=1, step="Building STFT config...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: building STFT config")
        stft_cfg = StftConfig(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            center=STFT_CENTER,
            window=WINDOW,
        )
        crop_cfg = CropConfig(time_frames=args.time_frames)
        setup_progress.update(
            setup_task, advance=1, step="Loading datasets...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: loading datasets")
        _train_ds, _val_ds, train_loader, val_loader = build_dataloaders(
            args, stft_cfg, crop_cfg, device
        )
        setup_progress.update(
            setup_task, advance=1, step="Initializing model...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing model")
        model = UNet2D(stems=len(STEMS_4), base_channels=args.base_channels).to(device)
        setup_progress.update(
            setup_task, advance=1, step="Initializing optimizer...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        setup_progress.update(
            setup_task, advance=1, step="Initializing scheduler...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        setup_progress.update(
            setup_task, advance=1, step="Initializing amp...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing amp")
        use_amp = bool(args.amp) and (device.type == "cuda")
        scaler = _make_grad_scaler(device, use_amp)

        start_epoch = 0
        global_step = 0

        setup_progress.update(
            setup_task, advance=1, step="Restoring checkpoint...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: restoring checkpoint (if requested)")
        if args.resume.strip():
            ckpt_path = args.resume.strip()
            start_epoch, global_step, extra = load_checkpoint(
                ckpt_path,
                model,
                optimizer,
                map_location=device,
            )
            logger.info(
                "Resumed from %s: epoch=%d step=%d extra_keys=%s",
                ckpt_path,
                int(start_epoch),
                int(global_step),
                list(extra.keys()),
            )

        setup_progress.update(
            setup_task, advance=1, step="Building run config...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: building run config")
        config = build_run_config(args, device, stft_cfg, crop_cfg)
        if wandb.run is not None:
            wandb.config.update(config)

        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        setup_progress.update(setup_task, advance=1, step="▁▁▁▁▁", step_style=BAR_FINISHED_STYLE)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        loss_dict, train_batches = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            epoch_index=epoch,
            clip_grad_norm=args.clip_grad_norm,
            stft_loss_weight=args.stft_loss_weight,
            sisdr_loss_weight=args.sisdr_loss_weight,
            stft_cfg=stft_cfg,
        )
        global_step += train_batches

        val_loss, _val_batches = eval_one_epoch(model, val_loader, device, epoch_index=epoch)

        scheduler.step(val_loss)

        dt = time.time() - t0
        lr_now = float(optimizer.param_groups[0].get("lr", args.lr))
        logger.info(
            "Epoch %d/%d - lr=%.6e kl=%.6f stft=%.6f sisdr=%.6f total=%.6f val_kl=%.6f - %.1fs",
            int(epoch + 1),
            int(args.epochs),
            float(lr_now),
            float(loss_dict["kl"]),
            float(loss_dict["stft"]),
            float(loss_dict["sisdr"]),
            float(loss_dict["total"]),
            float(val_loss),
            float(dt),
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "train/kl_loss": loss_dict["kl"],
                    "train/stft_loss": loss_dict["stft"],
                    "train/sisdr_loss": loss_dict["sisdr"],
                    "train/total_loss": loss_dict["total"],
                    "val/kl_loss": val_loss,
                    "train/lr": lr_now,
                    "time/epoch_time": dt,
                },
                step=epoch + 1,
            )

        do_save = ((epoch + 1) % args.save_every_epochs) == 0
        saved_ckpt = "-"
        saved_ts = "-"
        if do_save:
            ckpt_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pth"
            extra = {"config": config}
            save_checkpoint(
                str(ckpt_path),
                model,
                optimizer,
                epoch=epoch + 1,
                step=global_step,
                extra=extra,
            )
            saved_ckpt = str(ckpt_path)
            logger.info("Saved checkpoint: %s", str(ckpt_path))

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                saved_ts = str(ts_path)
                logger.info("Exported TorchScript: %s", str(ts_path))

        summary_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss_dict["total"]),
                "val_loss": float(val_loss),
                "checkpoint_path": saved_ckpt,
                "torchscript_path": saved_ts,
            }
        )

    print_training_summary(summary_rows, include_torchscript=bool(args.export_ts))


if __name__ == "__main__":
    raise SystemExit(main())
