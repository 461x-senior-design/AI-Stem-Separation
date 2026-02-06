import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.unet_2d import UNet2D
from src.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from src.training.musdb18hq_dataset import STEMS_4, CropConfig, Musdb18HQDataset, StftConfig


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
    p.add_argument("--num-workers", type=int, default=2)

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
    p.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="TensorBoard log directory.",
    )
    p.add_argument("--device", type=str, default="", help="cpu, cuda, or leave empty for auto.")

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


def main():
    args = parse_args()

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = UNet2D(stems=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stems = STEMS_4
    grad_hist_names = {"enc1.conv1.weight", "out_conv.weight"}

    start_epoch = 0
    global_step = 0

    if args.resume.strip():
        ckpt_path = args.resume.strip()
        start_epoch, global_step, extra = load_checkpoint(
            ckpt_path, model, optimizer, map_location=device
        )
        print(
            f"Resumed from {ckpt_path}: epoch={start_epoch}, step={global_step}, "
            f"extra_keys={list(extra.keys())}"
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

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard: initialize writer for this run.
    writer = SummaryWriter(log_dir=args.log_dir)
    # TensorBoard: log run configuration for traceability.
    writer.add_text("run/config", str(config))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        batches = 0
        example = None
        stem_loss_train = {s: 0.0 for s in stems}
        stem_stats_train = {
            s: {"sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf"), "count": 0}
            for s in stems
        }

        for mix_norm, targets_norm in train_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Model predicts per-stem ratio masks in [0,1]
            # (targets: stem_mag / (mix_mag + eps), clamped).
            pred_norm = model(mix_norm)  # [B, S, F, T]
            loss = l1_loss(pred_norm, targets_norm)

            loss.backward()
            optimizer.step()

            val = float(loss.detach().cpu().item())
            running += val
            batches += 1
            global_step += 1
            if example is None:
                # TensorBoard: cache a single example for spectrogram visualization.
                example = {
                    "mix": mix_norm[0].detach().cpu(),  # [1,F,T]
                    "pred_vocals": pred_norm[0, 2].detach().cpu(),  # [F,T]
                    "target_vocals": targets_norm[0, 2].detach().cpu(),  # [F,T]
                }
            # TensorBoard: per-stem loss accumulation (train).
            abs_diff = torch.abs(pred_norm - targets_norm)
            for i, stem in enumerate(stems):
                stem_loss_train[stem] += float(abs_diff[:, i].mean().detach().cpu().item())
            # TensorBoard: per-stem mask statistics (train).
            for i, stem in enumerate(stems):
                vals = pred_norm[:, i]
                stem_stats_train[stem]["sum"] += float(vals.sum().detach().cpu().item())
                stem_stats_train[stem]["sumsq"] += float(
                    (vals * vals).sum().detach().cpu().item()
                )
                stem_stats_train[stem]["min"] = min(
                    stem_stats_train[stem]["min"], float(vals.min().detach().cpu().item())
                )
                stem_stats_train[stem]["max"] = max(
                    stem_stats_train[stem]["max"], float(vals.max().detach().cpu().item())
                )
                stem_stats_train[stem]["count"] += int(vals.numel())

        train_loss = running / max(1, batches)

        model.eval()
        v_running = 0.0
        v_batches = 0
        stem_loss_val = {s: 0.0 for s in stems}
        stem_stats_val = {
            s: {"sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf"), "count": 0}
            for s in stems
        }
        with torch.no_grad():
            for mix_norm, targets_norm in val_loader:
                mix_norm = mix_norm.to(device, non_blocking=True)
                targets_norm = targets_norm.to(device, non_blocking=True)
                pred_norm = model(mix_norm)
                v_loss = l1_loss(pred_norm, targets_norm)
                v_running += float(v_loss.cpu().item())
                v_batches += 1
                # TensorBoard: per-stem loss accumulation (val).
                v_abs_diff = torch.abs(pred_norm - targets_norm)
                for i, stem in enumerate(stems):
                    stem_loss_val[stem] += float(
                        v_abs_diff[:, i].mean().detach().cpu().item()
                    )
                # TensorBoard: per-stem mask statistics (val).
                for i, stem in enumerate(stems):
                    vals = pred_norm[:, i]
                    stem_stats_val[stem]["sum"] += float(vals.sum().detach().cpu().item())
                    stem_stats_val[stem]["sumsq"] += float(
                        (vals * vals).sum().detach().cpu().item()
                    )
                    stem_stats_val[stem]["min"] = min(
                        stem_stats_val[stem]["min"], float(vals.min().detach().cpu().item())
                    )
                    stem_stats_val[stem]["max"] = max(
                        stem_stats_val[stem]["max"], float(vals.max().detach().cpu().item())
                    )
                    stem_stats_val[stem]["count"] += int(vals.numel())

        val_loss = v_running / max(1, v_batches)
        dt = time.time() - t0

        # TensorBoard: epoch-level scalars.
        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("loss/val", val_loss, epoch + 1)
        writer.add_scalar("time/epoch_sec", dt, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        # TensorBoard: per-stem loss and mask statistics (train/val).
        for stem in stems:
            writer.add_scalar(
                f"loss/train_{stem}", stem_loss_train[stem] / max(1, batches), epoch + 1
            )
            writer.add_scalar(
                f"loss/val_{stem}", stem_loss_val[stem] / max(1, v_batches), epoch + 1
            )
            for prefix, stats in [("train", stem_stats_train[stem]), ("val", stem_stats_val[stem])]:
                if stats["count"] > 0:
                    mean = stats["sum"] / stats["count"]
                    var = max((stats["sumsq"] / stats["count"]) - (mean * mean), 0.0)
                    std = var ** 0.5
                    writer.add_scalar(f"mask/{prefix}_{stem}_mean", mean, epoch + 1)
                    writer.add_scalar(f"mask/{prefix}_{stem}_std", std, epoch + 1)
                    writer.add_scalar(f"mask/{prefix}_{stem}_min", stats["min"], epoch + 1)
                    writer.add_scalar(f"mask/{prefix}_{stem}_max", stats["max"], epoch + 1)
        if example is not None:
            # TensorBoard: example spectrograms (mix, target, prediction).
            writer.add_image(
                "spec/mix",
                example["mix"].permute(0, 2, 1),
                epoch + 1,
                dataformats="CHW",
            )
            writer.add_image(
                "spec/target_vocals",
                example["target_vocals"].unsqueeze(0).transpose(1, 2),
                epoch + 1,
                dataformats="CHW",
            )
            writer.add_image(
                "spec/pred_vocals",
                example["pred_vocals"].unsqueeze(0).transpose(1, 2),
                epoch + 1,
                dataformats="CHW",
            )
        # TensorBoard: histogram snapshots for a few key layers.
        for name, param in model.named_parameters():
            if name in grad_hist_names:
                writer.add_histogram(
                    f"weights/{name}", param.detach().cpu(), epoch + 1
                )
                if param.grad is not None:
                    writer.add_histogram(
                        f"grads/{name}", param.grad.detach().cpu(), epoch + 1
                    )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
            f"time={dt:.1f}s"
        )

        do_save = ((epoch + 1) % args.save_every_epochs) == 0
        if do_save:
            ckpt_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pth"
            extra = {"config": config}
            save_checkpoint(
                str(ckpt_path), model, optimizer, epoch=epoch + 1, step=global_step, extra=extra
            )
            print(f"Saved checkpoint: {ckpt_path}")

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                print(f"Exported TorchScript: {ts_path}")

    # TensorBoard: flush and close.
    writer.close()


if __name__ == "__main__":
    main()
