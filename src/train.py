import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from src.models.unet_2d import UNet2D
from src.training.musdb18hq_dataset import Musdb18HQDataset, StftConfig, CropConfig, STEMS_4
from src.training.checkpointing import save_checkpoint, load_checkpoint, export_torchscript


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

    start_epoch = 0
    global_step = 0

    if args.resume.strip():
        ckpt_path = args.resume.strip()
        start_epoch, global_step, extra = load_checkpoint(
            ckpt_path, model, optimizer, map_location=device
        )
        print(
            f"Resumed from {ckpt_path}: epoch={start_epoch}, step={global_step}, extra_keys={list(extra.keys())}"
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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        batches = 0

        for mix_norm, targets_norm in train_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Model predicts per-stem ratio masks in [0,1] (targets: stem_mag / (mix_mag + eps), clamped).
            pred_norm = model(mix_norm)  # [B, S, F, T]
            loss = l1_loss(pred_norm, targets_norm)

            loss.backward()
            optimizer.step()

            val = float(loss.detach().cpu().item())
            running += val
            batches += 1
            global_step += 1

        train_loss = running / max(1, batches)

        model.eval()
        v_running = 0.0
        v_batches = 0
        with torch.no_grad():
            for mix_norm, targets_norm in val_loader:
                mix_norm = mix_norm.to(device, non_blocking=True)
                targets_norm = targets_norm.to(device, non_blocking=True)
                pred_norm = model(mix_norm)
                v_loss = l1_loss(pred_norm, targets_norm)
                v_running += float(v_loss.cpu().item())
                v_batches += 1

        val_loss = v_running / max(1, v_batches)
        dt = time.time() - t0

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


if __name__ == "__main__":
    main()
