# src/stemmy/train.py
"""Training entry point for the spectrogram-mask U-Net on MUSDB18-HQ.

This script trains a UNet2D to predict per-stem ratio masks from a normalized mono
mixture magnitude spectrogram.

Key invariants:
- STFT configuration is centralized in stemmy.constants.
- The STFT `center` setting (stemmy.constants.STFT_CENTER) must match across
  preprocessing, training, inference, and reconstruction.
- Dataset crops are chosen so each example produces exactly T time frames.

Performance optimizations:
- Automatic Mixed Precision (AMP) on CUDA for ~20-30% faster training
- torch.compile() for graph-level optimizations on PyTorch 2.0+
- Gradient clipping for training stability
- Persistent DataLoader workers to avoid respawn overhead
- Configurable num_workers for I/O parallelism

Outputs:
- Checkpoints (*.pth) written to --checkpoint-dir, containing model+optimizer state
  and a small config dict for reproducibility.
- Optional TorchScript export (*.pt) via --export-ts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stemmy.constants import (
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STEMS_4,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    WINDOW,
)
from stemmy.logging_config import get_logger, setup_logging
from stemmy.models.unet_2d import UNet2D
from stemmy.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from stemmy.training.musdb18hq_dataset import CropConfig, Musdb18HQDataset
from stemmy.training.stft import StftConfig

logger = get_logger(__name__)

# Local constant: used only by this training entry point unless you choose to share it.
_MASK_RENORM_EPS: float = 1e-8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train U-Net on MUSDB18-HQ (baseline).")

    p.add_argument(
        "--data-root",
        type=str,
        default="/home/ryan/shared/46x/stems/musdb18hq",
        help="MUSDB18-HQ root directory containing train/ and test/.",
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

    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")

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

    # Performance flags
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP) even on CUDA.",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile() model optimization.",
    )
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable).",
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    return p.parse_args()


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


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss used for mask regression."""
    return torch.mean(torch.abs(pred - target))


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
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be >= 0")

    if not isinstance(args.data_root, str) or args.data_root.strip() == "":
        raise ValueError("--data-root must be a non-empty string")
    if not isinstance(args.checkpoint_dir, str) or args.checkpoint_dir.strip() == "":
        raise ValueError("--checkpoint-dir must be a non-empty string")

    if args.resume.strip():
        resume_p = Path(args.resume.strip()).expanduser()
        if not resume_p.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_p}")
        if resume_p.is_dir():
            raise IsADirectoryError(
                f"--resume must point to a .pth file, got directory: {resume_p}"
            )


def build_dataloaders(
    args: argparse.Namespace,
    stft_cfg: StftConfig,
    crop_cfg: CropConfig,
    device: torch.device,
) -> tuple[Musdb18HQDataset, Musdb18HQDataset, DataLoader, DataLoader]:
    """Build train/val datasets and DataLoaders."""
    max_tracks = args.max_tracks if args.max_tracks > 0 else None

    logger.info(
        "Building datasets: data_root=%s, max_tracks=%s, time_frames=%d",
        args.data_root, max_tracks, crop_cfg.time_frames,
    )

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
    )

    logger.info(
        "Dataset sizes: train=%d tracks, val=%d tracks", len(train_ds), len(val_ds),
    )

    train_drop_last = len(train_ds) >= args.batch_size
    if not train_drop_last:
        logger.warning(
            "Train dataset size (%d) is smaller than batch_size (%d); "
            "disabling drop_last to avoid zero training batches.",
            len(train_ds), args.batch_size,
        )

    # Use persistent_workers when num_workers > 0 to avoid worker respawn overhead
    use_persistent = args.num_workers > 0
    val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=train_drop_last,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=use_persistent,
    )

    logger.info(
        "DataLoaders: batch_size=%d, num_workers=%d (train) / %d (val), "
        "persistent_workers=%s, pin_memory=%s",
        args.batch_size, args.num_workers, val_workers, use_persistent,
        device.type == "cuda",
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
        "mask_renorm_eps": _MASK_RENORM_EPS,
    }


def _renorm_sum_to_one(pred_norm: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize predicted masks so they sum to one across the stem dimension."""
    denom = pred_norm.sum(dim=1, keepdim=True) + float(eps)
    return pred_norm / denom


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float = 0.0,
) -> tuple[float, int]:
    """Run one training epoch and return (mean_loss, num_batches).

    Args:
        model: The model to train.
        optimizer: The optimizer.
        train_loader: Training DataLoader.
        device: Target device.
        use_amp: Whether to use Automatic Mixed Precision.
        scaler: GradScaler for AMP (required if use_amp is True on CUDA).
        grad_clip_norm: Max gradient norm for clipping (0 to disable).
    """
    model.train()
    running = 0.0
    batches = 0

    for mix_norm, targets_norm in train_loader:
        mix_norm = mix_norm.to(device, non_blocking=True)
        targets_norm = targets_norm.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use AMP autocast for forward pass + loss computation
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred_norm = model(mix_norm)
            pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
            pred_norm = _renorm_sum_to_one(pred_norm, eps=_MASK_RENORM_EPS)
            loss = l1_loss(pred_norm, targets_norm)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        running += float(loss.detach().cpu().item())
        batches += 1

        if batches % 50 == 0:
            logger.debug(
                "  batch %d: loss=%.6f", batches, running / batches,
            )

    mean_loss = running / max(1, batches)
    return mean_loss, batches


def eval_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, int]:
    """Run validation and return (mean_loss, num_batches)."""
    model.eval()
    running = 0.0
    batches = 0

    with torch.no_grad():
        for mix_norm, targets_norm in val_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_norm = model(mix_norm)
                pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
                pred_norm = _renorm_sum_to_one(pred_norm, eps=_MASK_RENORM_EPS)
                v_loss = l1_loss(pred_norm, targets_norm)

            running += float(v_loss.cpu().item())
            batches += 1

    mean_loss = running / max(1, batches)
    return mean_loss, batches


def _try_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Attempt to compile the model with torch.compile() for faster execution.

    Returns the compiled model on success, or the original model if compilation
    is not available or fails (e.g., on Windows or older PyTorch).

    torch.compile() is lazy — it defers actual compilation to the first forward
    pass. If the Triton backend is missing (common on Windows), the error only
    surfaces mid-training. We detect this upfront by checking for Triton.
    """
    import sys

    if not hasattr(torch, "compile"):
        logger.info("torch.compile not available (PyTorch < 2.0); skipping compilation.")
        return model

    # torch.compile with the inductor backend requires Triton on CUDA.
    # On Windows, Triton is typically unavailable.
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401
        except ImportError:
            logger.info(
                "torch.compile skipped: Triton is not installed (required on Windows/CUDA). "
                "Install triton or use --no-compile."
            )
            return model

    try:
        compiled = torch.compile(model)
        logger.info("Model compiled successfully with torch.compile().")
        return compiled
    except Exception as e:
        logger.warning("torch.compile() failed, falling back to eager mode: %s", e)
        return model


def main() -> None:
    """Main training loop."""
    args = parse_args()

    # Setup logging before anything else
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("Starting training session")
    logger.info("=" * 60)

    validate_args(args)

    device = pick_device(args.device)
    logger.info("Device: %s", device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    # Determine AMP usage: enabled by default on CUDA, disabled on CPU
    use_amp = (device.type == "cuda") and (not args.no_amp)
    logger.info("Automatic Mixed Precision (AMP): %s", "enabled" if use_amp else "disabled")

    # GradScaler for AMP (only meaningful on CUDA)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    stft_cfg = StftConfig(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        center=STFT_CENTER,
        window=WINDOW,
    )
    crop_cfg = CropConfig(time_frames=args.time_frames)

    logger.info(
        "STFT config: n_fft=%d, hop=%d, win=%d, center=%s",
        stft_cfg.n_fft, stft_cfg.hop_length, stft_cfg.win_length, stft_cfg.center,
    )

    _train_ds, _val_ds, train_loader, val_loader = build_dataloaders(
        args, stft_cfg, crop_cfg, device
    )

    model = UNet2D(stems=len(STEMS_4), base_channels=args.base_channels).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: UNet2D (stems=%d, base_channels=%d) - %d params (%d trainable)",
        len(STEMS_4), args.base_channels, param_count, trainable_count,
    )

    # Optionally compile model for faster execution
    if not args.no_compile:
        model = _try_compile_model(model)
    else:
        logger.info("Model compilation disabled via --no-compile.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    logger.info(
        "Optimizer: AdamW (lr=%.2e, weight_decay=%.2e)", args.lr, args.weight_decay,
    )
    logger.info(
        "Scheduler: ReduceLROnPlateau (factor=%.2f, patience=%d, min_lr=%.2e)",
        args.lr_factor, args.lr_patience, args.min_lr,
    )
    if args.grad_clip_norm > 0:
        logger.info("Gradient clipping: max_norm=%.2f", args.grad_clip_norm)
    else:
        logger.info("Gradient clipping: disabled")

    start_epoch = 0
    global_step = 0

    if args.resume.strip():
        ckpt_path = args.resume.strip()
        start_epoch, global_step, extra = load_checkpoint(
            ckpt_path,
            model,
            optimizer,
            map_location=device,
        )
        logger.info(
            "Resumed from %s: epoch=%d, step=%d, extra_keys=%s",
            ckpt_path, start_epoch, global_step, list(extra.keys()),
        )

    config = build_run_config(args, device, stft_cfg, crop_cfg)

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training for %d epochs (starting from epoch %d)", args.epochs, start_epoch + 1)
    logger.info("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_batches = train_one_epoch(
            model, optimizer, train_loader, device,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip_norm=args.grad_clip_norm,
        )
        global_step += train_batches

        val_loss, _val_batches = eval_one_epoch(
            model, val_loader, device,
            use_amp=use_amp,
        )

        scheduler.step(val_loss)

        dt = time.time() - t0
        lr_now = float(optimizer.param_groups[0].get("lr", args.lr))

        logger.info(
            "Epoch %d/%d - lr=%.2e train_loss=%.6f val_loss=%.6f - %.1fs "
            "(%d batches, %.2fs/batch)",
            epoch + 1, args.epochs, lr_now, train_loss, val_loss,
            dt, train_batches, dt / max(1, train_batches),
        )

        do_save = ((epoch + 1) % args.save_every_epochs) == 0
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
            logger.info("Saved checkpoint: %s", ckpt_path)

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                logger.info("Exported TorchScript: %s", ts_path)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
