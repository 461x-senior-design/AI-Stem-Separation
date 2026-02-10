"""Legacy training entry point.

This script delegates to the main training entry point in stemmy.train.
For the full set of CLI arguments (including --no-amp, --no-compile,
--grad-clip-norm, --log-level), run:

    python -m stemmy.train --help
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import wandb
from stemmy.logging_config import get_logger, setup_logging
from stemmy.models.unet_2d import UNet2D
from stemmy.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from stemmy.training.musdb18hq_dataset import STEMS_4, CropConfig, Musdb18HQDataset, StftConfig
from wandb_config import wandb_run

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net on MUSDB18-HQ (baseline).")

    p.add_argument(
        "--data-root",
        type=str,
        default="/home/ryan/shared/46x/stems/musdb18hq",
        help="MUSDB18-HQ root directory containing train/ and test/.",
    )

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument(
        "--time-frames", type=int, default=256, help="Fixed STFT time frames T (e.g., 256 or 512)."
    )
    p.add_argument(
        "--max-tracks", type=int, default=0, help="If >0, limit number of tracks per split."
    )

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument(
        "--resume", type=str, default="", help="Path to a .pth checkpoint to resume from."
    )
    p.add_argument("--save-every-epochs", type=int, default=1)

    p.add_argument(
        "--export-ts",
        action="store_true",
        help="Export TorchScript .pt after each checkpoint save.",
    )
    p.add_argument("--device", type=str, default="", help="cpu, cuda, or leave empty for auto.")

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

    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    return p.parse_args()


def pick_device(arg):
    if arg.strip():
        if arg.strip() == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but CUDA is not available.")
        return torch.device(arg.strip())

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def _try_compile_model(model):
    """Attempt torch.compile(); returns original model on failure."""
    if not hasattr(torch, "compile"):
        logger.info("torch.compile not available; skipping.")
        return model
    try:
        compiled = torch.compile(model)
        logger.info("Model compiled with torch.compile().")
        return compiled
    except Exception as e:
        logger.warning("torch.compile() failed: %s", e)
        return model


@wandb_run(job_type="training", name="training")
def train(args):
    """Main training function wrapped with wandb decorator."""
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.time_frames not in [256, 512]:
        raise ValueError("--time-frames must be 256 or 512 for this baseline.")

    device = pick_device(args.device)
    logger.info("Device: %s", device)

    # AMP setup
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    logger.info("AMP: %s", "enabled" if use_amp else "disabled")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_mem / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    stft_cfg = StftConfig(sample_rate=44100, n_fft=4096, hop_length=1024, win_length=4096)
    crop_cfg = CropConfig(time_frames=args.time_frames)

    max_tracks = args.max_tracks if args.max_tracks > 0 else None

    train_ds = Musdb18HQDataset(
        root_dir=args.data_root,
        split="train",
        stft_cfg=stft_cfg,
        crop_cfg=crop_cfg,
        stems=STEMS_4,
        max_tracks=max_tracks,
        deterministic=False,
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
        seed=0,
    )

    logger.info("Dataset: train=%d, val=%d tracks", len(train_ds), len(val_ds))

    use_persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2) if args.num_workers > 0 else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=use_persistent,
    )

    logger.info(
        "DataLoaders: batch=%d, workers=%d, persistent=%s",
        args.batch_size, args.num_workers, use_persistent,
    )

    model = UNet2D(stems=4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", param_count)

    # Optionally compile
    if not args.no_compile:
        model = _try_compile_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0

    if args.resume.strip():
        ckpt_path = args.resume.strip()
        start_epoch, global_step, extra = load_checkpoint(
            ckpt_path, model, optimizer, map_location=device
        )
        logger.info(
            "Resumed from %s: epoch=%d, step=%d", ckpt_path, start_epoch, global_step,
        )

    config = {
        "data_root": args.data_root,
        "sample_rate": stft_cfg.sample_rate,
        "n_fft": stft_cfg.n_fft,
        "hop_length": stft_cfg.hop_length,
        "win_length": stft_cfg.win_length,
        "time_frames": crop_cfg.time_frames,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "stems": STEMS_4,
        "max_tracks": args.max_tracks,
        "center": False,
    }

    # Log config to wandb
    if wandb.run is not None:
        wandb.config.update(config)
        wandb.config.update(
            {
                "epochs": args.epochs,
                "num_workers": args.num_workers,
                "save_every_epochs": args.save_every_epochs,
                "amp": use_amp,
                "grad_clip_norm": args.grad_clip_norm,
            }
        )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.grad_clip_norm > 0:
        logger.info("Gradient clipping: max_norm=%.2f", args.grad_clip_norm)

    logger.info("Starting training: %d epochs", args.epochs)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        batches = 0

        for mix_norm, targets_norm in train_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_norm = model(mix_norm)  # [B, S, F, T]
                loss = l1_loss(pred_norm, targets_norm)

            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip_norm,
                    )
                optimizer.step()

            val = float(loss.detach().cpu().item())
            running += val
            batches += 1
            global_step += 1

            # Log per-step loss
            if wandb.run is not None:
                wandb.log({"train/loss_step": val}, step=global_step)

        train_loss = running / max(1, batches)

        model.eval()
        v_running = 0.0
        v_batches = 0
        with torch.no_grad():
            for mix_norm, targets_norm in val_loader:
                mix_norm = mix_norm.to(device, non_blocking=True)
                targets_norm = targets_norm.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred_norm = model(mix_norm)
                    v_loss = l1_loss(pred_norm, targets_norm)

                v_running += float(v_loss.cpu().item())
                v_batches += 1

        val_loss = v_running / max(1, v_batches)
        dt = time.time() - t0

        logger.info(
            "Epoch %d/%d | train_loss=%.6f val_loss=%.6f | "
            "%.1fs (%d batches, %.2fs/batch)",
            epoch + 1, args.epochs, train_loss, val_loss,
            dt, batches, dt / max(1, batches),
        )

        # Log per-epoch metrics
        if wandb.run is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch + 1,
                    "time/epoch_time": dt,
                },
                step=global_step,
            )

        do_save = ((epoch + 1) % args.save_every_epochs) == 0
        if do_save:
            ckpt_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pth"
            extra = {"config": config}
            save_checkpoint(
                str(ckpt_path), model, optimizer, epoch=epoch + 1, step=global_step, extra=extra
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                logger.info("Exported TorchScript: %s", ts_path)

    logger.info("Training complete.")


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("=" * 50)
    logger.info("Starting training (legacy entry point)")
    logger.info("=" * 50)
    train(args)


if __name__ == "__main__":
    main()
