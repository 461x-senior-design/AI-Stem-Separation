import argparse
import time
from pathlib import Path

import torch

from alive_progress import alive_bar
from alive_progress.animations.spinners import frame_spinner_factory
from torch.utils.data import DataLoader

from src.constants import (
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
)
from src.models.unet_2d import UNet2D
from src.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from src.training.musdb18hq_dataset import STEMS_4, CropConfig, Musdb18HQDataset, StftConfig


_DEFAULT_DUBSTEP_MIDI_TEXT = """
# Mini step-sequencer format:
# key=value headers
# track lines as "<name>: token token token ..."
# tokens: "."/"-" = rest, "x" = max velocity, "0..8" = velocity.
steps=16
subframes=6
#      1 . 2 . 3 . 4 . 5 . 6 . 7 . 8 .
kick:  8 8 . . . . 8 . 8 . . . . . 8 8
snare: . . . . 7 . . . . . . . 7 . . .
hat:   6 6 . 8 7 . 4 5 6 7 8 . 5 . 7 8
"""


def _make_eq_frames(midi_text=_DEFAULT_DUBSTEP_MIDI_TEXT, bars=1):
    """Create a long 5-band EQ loop from a small MIDI-like text pattern."""
    levels = ("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")

    def _to_cell(value):
        clamped = max(1, min(8, int(round(value))))
        return levels[clamped - 1]

    def _parse_step_token(token):
        token = token.strip().lower()
        if token in {".", "-", "_"}:
            return 0
        if token == "x":
            return 8
        value = int(token)
        if value < 0 or value > 8:
            raise ValueError(f"Velocity token must be 0..8, got '{token}'.")
        return value

    def _parse_midi_like_text(text):
        cfg = {"steps": 8, "subframes": 6}
        tracks = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line and ":" not in line:
                key, value = [chunk.strip().lower() for chunk in line.split("=", 1)]
                if key not in cfg:
                    raise ValueError(f"Unknown config key '{key}'.")
                cfg[key] = int(value)
                continue
            if ":" not in line:
                raise ValueError(f"Invalid pattern line: '{line}'.")
            name, body = [chunk.strip().lower() for chunk in line.split(":", 1)]
            tokens = [t for t in body.split() if t]
            tracks[name] = [_parse_step_token(token) for token in tokens]

        steps = cfg["steps"]
        if steps <= 0:
            raise ValueError("steps must be > 0.")
        if cfg["subframes"] <= 0:
            raise ValueError("subframes must be > 0.")

        for track_name in ("kick", "snare", "hat"):
            if track_name not in tracks:
                tracks[track_name] = [0] * steps
            if len(tracks[track_name]) != steps:
                raise ValueError(
                    f"Track '{track_name}' has {len(tracks[track_name])} steps, expected {steps}."
                )
        return cfg, tracks

    cfg, tracks = _parse_midi_like_text(midi_text)
    frames = []
    kick_env = {
        "low": (8, 7, 6, 5, 4, 3),
        "low_mid": (4, 4, 3, 2, 2, 1),
        "mid": (2, 2, 1, 1, 1, 1),
        "mid_high": (1, 1, 1, 1, 1, 1),
    }
    snare_env = {
        "low": (3, 2, 2, 1, 1, 1),
        "low_mid": (3, 4, 4, 3, 2, 1),
        "mid": (4, 6, 7, 6, 4, 3),
        "mid_high": (2, 3, 4, 3, 2, 1),
    }
    hat_env = {
        "mid_high": (2, 2, 1, 1, 1, 1),
        "high": (8, 7, 6, 5, 4, 3),
    }
    subframes = cfg["subframes"]

    for _ in range(bars):
        for step_idx in range(cfg["steps"]):
            kick_vel = tracks["kick"][step_idx] / 8.0
            snare_vel = tracks["snare"][step_idx] / 8.0
            hat_vel = tracks["hat"][step_idx] / 8.0

            for i in range(subframes):
                env_i = i % len(kick_env["low"])
                low = 1 + kick_env["low"][env_i] * kick_vel + snare_env["low"][env_i] * snare_vel
                low_mid = (
                    1
                    + kick_env["low_mid"][env_i] * kick_vel
                    + snare_env["low_mid"][env_i] * snare_vel
                )
                mid = 1 + kick_env["mid"][env_i] * kick_vel + snare_env["mid"][env_i] * snare_vel
                mid_high = (
                    1
                    + kick_env["mid_high"][env_i] * kick_vel
                    + snare_env["mid_high"][env_i] * snare_vel
                    + hat_env["mid_high"][env_i] * hat_vel
                )
                high = 1 + hat_env["high"][env_i] * hat_vel
                frames.append(
                    _to_cell(low)
                    + _to_cell(low_mid)
                    + _to_cell(mid)
                    + _to_cell(mid_high)
                    + _to_cell(high)
                )

    return tuple(frames)


DUBSTEP = frame_spinner_factory((_make_eq_frames(bars=1)))


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

    stft_cfg = StftConfig(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
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
        "center": STFT_CENTER,
        "waveform_norm": args.waveform_norm,
    }

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        batches = 0

        # Visualize per-batch progress and most recent loss.
        with alive_bar(
            total=len(train_loader),
            title=f"Epoch {epoch + 1} Train",
            dual_line=True,
            force_tty=True,
            refresh_secs=1 / 15,
            # Custom EQ spinner.
            spinner=DUBSTEP,
        ) as bar:
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
                # Update progress bar with latest batch loss.
                bar.text = f"loss={val:.6f}"
                bar()

        train_loss = running / max(1, batches)

        model.eval()
        v_running = 0.0
        v_batches = 0
        with torch.no_grad():
            # Separate bar for validation pass to track its batches and loss.
            with alive_bar(
                total=len(val_loader),
                title=f"Epoch {epoch + 1} Val  ",
                dual_line=True,
                force_tty=True,
                refresh_secs=1 / 15,
                spinner=DUBSTEP,
            ) as bar:
                for mix_norm, targets_norm in val_loader:
                    mix_norm = mix_norm.to(device, non_blocking=True)
                    targets_norm = targets_norm.to(device, non_blocking=True)
                    pred_norm = model(mix_norm)
                    pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
                    pred_norm = pred_norm / (pred_norm.sum(dim=1, keepdim=True) + 1e-8)
                    v_loss = l1_loss(pred_norm, targets_norm)
                    v_running += float(v_loss.cpu().item())
                    v_batches += 1
                    # Update progress bar with latest validation loss.
                    bar.text = f"loss={float(v_loss.cpu().item()):.6f}"
                    bar()

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
