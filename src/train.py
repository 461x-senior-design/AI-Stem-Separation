import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from torch.utils.data import DataLoader

from src.constants import (
    BAR_FINISHED_STYLE,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    LOSS_STYLE,
    N_FFT,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
)
from src.models.unet_2d import UNet2D
from src.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from src.training.musdb18hq_dataset import STEMS_4, CropConfig, Musdb18HQDataset, StftConfig
from src.ui.progress_theme import create_setup_progress, create_themed_progress, start_eq_animator


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
        "--waveform-norm",
        type=str,
        default=DEFAULT_WAVEFORM_NORM,
        choices=["peak", "rms", "none"],
        help="Waveform normalization method (applies to mix and stems).",
    )

    p.add_argument(
        "--time-frames", type=int, default=256, help="Fixed STFT time frames T (256 or 512)."
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


def print_training_summary(rows, include_torchscript=False):
    console = Console()
    table = Table(title="Training Summary", header_style="bold #B65CFF")
    table.add_column("Epoch", justify="right", style="white")
    table.add_column("Train Loss", justify="right", style="white")
    table.add_column("Val Loss", justify="right", style="white")
    table.add_column("Checkpoint", style="#B8A9D9")
    if include_torchscript:
        table.add_column("TorchScript", style="#39FF14")

    best_train = min((row["train_loss"] for row in rows), default=None)
    worst_train = max((row["train_loss"] for row in rows), default=None)
    best_val = min((row["val_loss"] for row in rows), default=None)
    worst_val = max((row["val_loss"] for row in rows), default=None)

    def style_loss(value, best, worst):
        if best is None or worst is None:
            return f"[white]{value:.6f}[/]"
        if value == best:
            return f"[#39FF14]{value:.6f}[/]"
        if value == worst:
            return f"[#FF4D6D]{value:.6f}[/]"
        return f"[white]{value:.6f}[/]"

    for row in rows:
        cells = [
            str(row["epoch"]),
            style_loss(row["train_loss"], best_train, worst_train),
            style_loss(row["val_loss"], best_val, worst_val),
            row["checkpoint_path"],
        ]
        if include_torchscript:
            cells.append(row["torchscript_path"])
        table.add_row(*cells)

    console.print()
    console.print(table)


def main():
    args = parse_args()
    summary_rows = []

    with create_setup_progress("Setup", title_style="bold #B65CFF") as setup_progress:
        setup_task = setup_progress.add_task(
            "setup", total=8, step="validate args", step_style=LOSS_STYLE
        )

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
        setup_progress.update(setup_task, advance=1, step="pick device", step_style=LOSS_STYLE)

        device = pick_device(args.device)
        setup_progress.update(
            setup_task, advance=1, step="build STFT configs", step_style=LOSS_STYLE
        )

        stft_cfg = StftConfig(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        crop_cfg = CropConfig(time_frames=args.time_frames)
        setup_progress.update(setup_task, advance=1, step="load datasets", step_style=LOSS_STYLE)

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
            seed=0,
        )
        setup_progress.update(
            setup_task, advance=1, step="build dataloaders", step_style=LOSS_STYLE
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
        setup_progress.update(
            setup_task, advance=1, step="init model and optimizer", step_style=LOSS_STYLE
        )

        model = UNet2D(stems=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        setup_progress.update(
            setup_task, advance=1, step="restore checkpoint", step_style=LOSS_STYLE
        )

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
        setup_progress.update(
            setup_task, advance=1, step="finalize run config", step_style=LOSS_STYLE
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
            "center": STFT_CENTER,
            "waveform_norm": args.waveform_norm,
        }

        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        setup_progress.update(setup_task, advance=1, step="▁▁▁▁▁", step_style=BAR_FINISHED_STYLE)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # t0 = time.time()
        running = 0.0
        batches = 0

        # Rich progress layout with a green EQ lane and live loss updates.
        with create_themed_progress(
            "Epoch {task.fields[epoch]} Train", title_style="bold #B65CFF"
        ) as progress:
            train_task = progress.add_task(
                "train",
                total=len(train_loader),
                epoch=epoch + 1,
                eq="",
            )
            train_anim_stop, train_anim_thread = start_eq_animator(progress, train_task, fps=8)
            try:
                for mix_norm, targets_norm in train_loader:
                    mix_norm = mix_norm.to(device, non_blocking=True)
                    targets_norm = targets_norm.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    # Model predicts per-stem ratio masks in [0,1]
                    # (targets: stem_mag / (mix_mag + eps), clamped, then sum-to-1).
                    pred_norm = model(mix_norm)  # [B, S, F, T]
                    pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
                    pred_norm = pred_norm / (pred_norm.sum(dim=1, keepdim=True) + 1e-8)
                    loss = l1_loss(pred_norm, targets_norm)

                    loss.backward()
                    optimizer.step()

                    val = float(loss.detach().cpu().item())
                    running += val
                    batches += 1
                    global_step += 1
                    # Batch updates only advance the bar; EQ runs in animator thread.
                    progress.update(train_task, advance=1)
            finally:
                train_anim_stop.set()
                train_anim_thread.join()
                progress.update(train_task, eq="▁▁▁▁▁")

        train_loss = running / max(1, batches)

        model.eval()
        v_running = 0.0
        v_batches = 0
        with torch.no_grad():
            # Validation uses the same EQ animation with a separate task/column title style.
            with create_themed_progress(
                "Epoch {task.fields[epoch]} Val  ", title_style="bold #9D4EDD"
            ) as progress:
                val_task = progress.add_task(
                    "val",
                    total=len(val_loader),
                    epoch=epoch + 1,
                    eq="",
                )
                val_anim_stop, val_anim_thread = start_eq_animator(progress, val_task, fps=8)
                try:
                    for mix_norm, targets_norm in val_loader:
                        mix_norm = mix_norm.to(device, non_blocking=True)
                        targets_norm = targets_norm.to(device, non_blocking=True)
                        pred_norm = model(mix_norm)
                        pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
                        pred_norm = pred_norm / (pred_norm.sum(dim=1, keepdim=True) + 1e-8)
                        v_loss = l1_loss(pred_norm, targets_norm)
                        v_loss_value = float(v_loss.cpu().item())
                        v_running += v_loss_value
                        v_batches += 1
                        # Validation updates only advance the bar; EQ runs independently.
                        progress.update(val_task, advance=1)
                finally:
                    val_anim_stop.set()
                    val_anim_thread.join()
                    progress.update(val_task, eq="▁▁▁▁▁")

        val_loss = v_running / max(1, v_batches)

        do_save = ((epoch + 1) % args.save_every_epochs) == 0
        saved_ckpt = "-"
        saved_ts = "-"
        if do_save:
            ckpt_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pth"
            extra = {"config": config}
            save_checkpoint(
                str(ckpt_path), model, optimizer, epoch=epoch + 1, step=global_step, extra=extra
            )
            saved_ckpt = str(ckpt_path)

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                saved_ts = str(ts_path)

        summary_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint_path": saved_ckpt,
                "torchscript_path": saved_ts,
            }
        )

    print_training_summary(summary_rows, include_torchscript=args.export_ts)


if __name__ == "__main__":
    main()
