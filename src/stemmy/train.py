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
"""

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
from stemmy.models.unet_2d import UNet2D
from stemmy.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from stemmy.training.musdb18hq_dataset import CropConfig, Musdb18HQDataset
from stemmy.training.stft import StftConfig

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

    train_drop_last = len(train_ds) >= args.batch_size
    if not train_drop_last:
        print(
            f"WARNING: train dataset size ({len(train_ds)}) is smaller than batch_size "
            f"({args.batch_size}); disabling drop_last to avoid zero training batches."
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
) -> tuple[float, int]:
    """Run one training epoch and return (mean_loss, num_batches)."""
    model.train()
    running = 0.0
    batches = 0

    for mix_norm, targets_norm in train_loader:
        mix_norm = mix_norm.to(device, non_blocking=True)
        targets_norm = targets_norm.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred_norm = model(mix_norm)
        pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
        pred_norm = _renorm_sum_to_one(pred_norm, eps=_MASK_RENORM_EPS)

        loss = l1_loss(pred_norm, targets_norm)
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item())
        batches += 1

    mean_loss = running / max(1, batches)
    return mean_loss, batches


def eval_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    """Run validation and return (mean_loss, num_batches)."""
    model.eval()
    running = 0.0
    batches = 0

    with torch.no_grad():
        for mix_norm, targets_norm in val_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            pred_norm = model(mix_norm)
            pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
            pred_norm = _renorm_sum_to_one(pred_norm, eps=_MASK_RENORM_EPS)

            v_loss = l1_loss(pred_norm, targets_norm)
            running += float(v_loss.cpu().item())
            batches += 1

    mean_loss = running / max(1, batches)
    return mean_loss, batches


def main() -> None:
    """Main training loop."""
    args = parse_args()
    validate_args(args)

    device = pick_device(args.device)

    stft_cfg = StftConfig(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        center=STFT_CENTER,
        window=WINDOW,
    )
    crop_cfg = CropConfig(time_frames=args.time_frames)

    _train_ds, _val_ds, train_loader, val_loader = build_dataloaders(
        args, stft_cfg, crop_cfg, device
    )

    model = UNet2D(stems=len(STEMS_4), base_channels=args.base_channels).to(device)

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
        print(
            f"Resumed from {ckpt_path}: epoch={start_epoch}, step={global_step}, "
            f"extra_keys={list(extra.keys())}"
        )

    config = build_run_config(args, device, stft_cfg, crop_cfg)

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_batches = train_one_epoch(model, optimizer, train_loader, device)
        global_step += train_batches

        val_loss, _val_batches = eval_one_epoch(model, val_loader, device)

        scheduler.step(val_loss)

        dt = time.time() - t0
        lr_now = float(optimizer.param_groups[0].get("lr", args.lr))
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"lr={lr_now:.6e} train_loss={train_loss:.6f} val_loss={val_loss:.6f} - "
            f"time={dt:.1f}s"
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
            print(f"Saved checkpoint: {ckpt_path}")

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                print(f"Exported TorchScript: {ts_path}")


if __name__ == "__main__":
    main()
