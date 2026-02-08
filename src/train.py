import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from torch.utils.data import DataLoader

from src.constants import (
    BAR_COMPLETE_STYLE,
    BAR_FINISHED_STYLE,
    DEFAULT_WAVEFORM_NORM,
    FINAL_AVG_LOSS_STYLE,
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


def _add_checkpoint_nodes(parent: Tree, path: Path):
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        if entry.is_dir():
            node = parent.add(f"{entry.name}/", style="#B8A9D9")
            _add_checkpoint_nodes(node, entry)
        else:
            parent.add(entry.name, style="white")


def print_checkpoint_tree(ckpt_dir: Path):
    console = Console()
    title = Text()
    title.append("\nCheckpoint Directory:", style="bold white")
    title.append(f" {ckpt_dir}", style="white")
    root = Tree(title, style="white", guide_style="#B8A9D9")
    if not ckpt_dir.exists():
        root.add("(missing)", style="#FF4D6D")
    else:
        _add_checkpoint_nodes(root, ckpt_dir)
    console.print(root)


def main():
    args = parse_args()

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
                loss="--",
                loss_label="    Loss:",
                loss_style=LOSS_STYLE,
            )
            train_anim_stop, train_anim_thread = start_eq_animator(progress, train_task, fps=8)
            try:
                train_best = float("inf")
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
                    # Batch updates advance progress and loss only; EQ runs in animator thread.
                    progress.update(
                        train_task,
                        advance=1,
                        loss=f"{val:.6f}",
                        loss_label="    Loss:",
                        loss_style=(BAR_FINISHED_STYLE if val < train_best else BAR_COMPLETE_STYLE),
                    )
                    train_best = min(train_best, val)
                # Replace last-batch loss with epoch-average loss before closing the bar.
                progress.update(
                    train_task,
                    loss=f"{(running / max(1, batches)):.6f}",
                    loss_label="AVG Loss:",
                    loss_style=FINAL_AVG_LOSS_STYLE,
                )
            finally:
                train_anim_stop.set()
                train_anim_thread.join()
                progress.update(train_task, eq="▁▁▁▁▁")

        # train_loss = running / max(1, batches)

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
                    loss="--",
                    loss_label="    Loss:",
                    loss_style=LOSS_STYLE,
                )
                val_anim_stop, val_anim_thread = start_eq_animator(progress, val_task, fps=8)
                try:
                    val_best = float("inf")
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
                        # Validation updates advance and loss only; EQ runs independently.
                        progress.update(
                            val_task,
                            advance=1,
                            loss=f"{v_loss_value:.6f}",
                            loss_label="    Loss:",
                            loss_style=(
                                BAR_FINISHED_STYLE
                                if v_loss_value < val_best
                                else BAR_COMPLETE_STYLE
                            ),
                        )
                        val_best = min(val_best, v_loss_value)
                    # Replace last-batch loss with epoch-average val loss before closing the bar.
                    progress.update(
                        val_task,
                        loss=f"{(v_running / max(1, v_batches)):.6f}",
                        loss_label="AVG Loss:",
                        loss_style=FINAL_AVG_LOSS_STYLE,
                    )
                finally:
                    val_anim_stop.set()
                    val_anim_thread.join()
                    progress.update(val_task, eq="▁▁▁▁▁")

        # val_loss = v_running / max(1, v_batches)
        # dt = time.time() - t0
        #
        # print(
        #     f"Epoch {epoch + 1}/{args.epochs} | "
        #     f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
        #     f"time={dt:.1f}s"
        # )

        do_save = ((epoch + 1) % args.save_every_epochs) == 0
        if do_save:
            ckpt_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pth"
            extra = {"config": config}
            save_checkpoint(
                str(ckpt_path), model, optimizer, epoch=epoch + 1, step=global_step, extra=extra
            )

            if args.export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                print(f"Exported TorchScript: {ts_path}")

    print_checkpoint_tree(ckpt_dir)


if __name__ == "__main__":
    main()
