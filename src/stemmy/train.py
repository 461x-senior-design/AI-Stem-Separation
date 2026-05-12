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

Training changes in this version:
- Model outputs are treated as logits over stems.
- Predicted masks are computed with softmax across stems (sum-to-one, differentiable).
- Loss is KL divergence between target distribution and predicted distribution.
"""

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dotenv import dotenv_values
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from stemmy.constants import (
    BAR_FINISHED_STYLE,
    BOLD_PURPLE,
    BOLD_VIOLET,
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
from stemmy.tool.click_options import add_options, processing_options, training_options
from stemmy.tool.progress_theme import (
    create_setup_progress,
    create_themed_progress,
    start_eq_animator,
)
from stemmy.training.checkpointing import export_torchscript, load_checkpoint, save_checkpoint
from stemmy.training.musdb18hq_dataset import CropConfig, Musdb18HQDataset
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
    """Unused placeholder to preserve module compatibility."""
    raise NotImplementedError("Use the Click command entry point instead.")


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
    raise NotImplementedError("Argument validation is handled in main().")


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
    data_root: str,
    max_tracks: int,
    waveform_norm: str,
    spectrogram_norm: str,
    batch_size: int,
    num_workers: int,
    stft_cfg: StftConfig,
    crop_cfg: CropConfig,
    device: torch.device,
) -> tuple[Musdb18HQDataset, Musdb18HQDataset, DataLoader, DataLoader]:
    """Build train/val datasets and DataLoaders."""
    effective_max_tracks = max_tracks if max_tracks > 0 else None

    train_ds = Musdb18HQDataset(
        root_dir=data_root,
        split="train",
        stft_cfg=stft_cfg,
        crop_cfg=crop_cfg,
        stems=STEMS_4,
        max_tracks=effective_max_tracks,
        deterministic=False,
        waveform_norm=waveform_norm,
        spectrogram_norm=spectrogram_norm,
        seed=0,
    )
    val_ds = Musdb18HQDataset(
        root_dir=data_root,
        split="test",
        stft_cfg=stft_cfg,
        crop_cfg=crop_cfg,
        stems=STEMS_4,
        max_tracks=effective_max_tracks,
        deterministic=True,
        waveform_norm=waveform_norm,
        spectrogram_norm=spectrogram_norm,
        seed=0,
    )

    train_drop_last = len(train_ds) >= batch_size
    if not train_drop_last:
        logger.warning(
            "Train dataset size (%d) is smaller than batch_size (%d); disabling drop_last "
            "to avoid zero training batches.",
            int(len(train_ds)),
            int(batch_size),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=train_drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_run_config(
    data_root: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    base_channels: int,
    lr_factor: float,
    lr_patience: int,
    min_lr: float,
    max_tracks: int,
    waveform_norm: str,
    spectrogram_norm: str,
    seed: int,
    amp: bool,
    device: torch.device,
    stft_cfg: StftConfig,
    crop_cfg: CropConfig,
) -> dict:
    """Build a small config dict stored inside checkpoints for reproducibility."""
    return {
        "data_root": data_root,
        "sample_rate": stft_cfg.sample_rate,
        "n_fft": stft_cfg.n_fft,
        "hop_length": stft_cfg.hop_length,
        "win_length": stft_cfg.win_length,
        "center": bool(stft_cfg.center),
        "window": stft_cfg.window,
        "time_frames": crop_cfg.time_frames,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "base_channels": base_channels,
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
        "min_lr": min_lr,
        "device": str(device),
        "stems": STEMS_4,
        "max_tracks": max_tracks,
        "waveform_norm": waveform_norm,
        "spectrogram_norm": spectrogram_norm,
        "loss": "kl_div(target||pred)",
        "pred_activation": "softmax_over_stems",
        "target_mask_mode": "stem_mag / sum_stem_mags",
        "seed": int(seed),
        "amp": bool(amp),
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


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    scaler: Any,
    epoch_index: int,
) -> tuple[float, int]:
    """Run one training epoch and return (mean_loss, num_batches)."""
    model.train()
    running = 0.0
    batches = 0

    if _progress_disabled():
        logger.info("Epoch %d train: %d batches", int(epoch_index + 1), int(len(train_loader)))
        for mix_norm, targets_norm in train_loader:
            mix_norm = mix_norm.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with _autocast_context(device, use_amp):
                    logits = model(mix_norm)
                    loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(mix_norm)
                loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)
                loss.backward()
                optimizer.step()

            running += float(loss.detach().cpu().item())
            batches += 1
        mean_loss = running / max(1, batches)
        return mean_loss, batches

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
            for mix_norm, targets_norm in train_loader:
                mix_norm = mix_norm.to(device, non_blocking=True)
                targets_norm = targets_norm.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with _autocast_context(device, use_amp):
                        logits = model(mix_norm)
                        loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(mix_norm)
                    loss = kl_div_loss_from_logits(logits, targets_norm, eps=_LOSS_EPS)
                    loss.backward()
                    optimizer.step()

                running += float(loss.detach().cpu().item())
                batches += 1
                progress.update(train_task, advance=1)
        finally:
            train_anim_stop.set()
            train_anim_thread.join()
            progress.update(train_task, eq="▁▁▁▁▁")

    mean_loss = running / max(1, batches)
    return mean_loss, batches


def eval_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch_index: int,
) -> tuple[float, int]:
    """Run validation and return (mean_loss, num_batches)."""
    model.eval()
    running = 0.0
    batches = 0

    if _progress_disabled():
        logger.info("Epoch %d val: %d batches", int(epoch_index + 1), int(len(val_loader)))
        with torch.no_grad():
            for mix_norm, targets_norm in val_loader:
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
                for mix_norm, targets_norm in val_loader:
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


@click.command("train")
@add_options(processing_options)
@add_options(training_options)
@wandb_run(job_type="training", name="training")
def main(
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    waveform_norm: str,
    spectrogram_norm: str,
    time_frames: int,
    max_tracks: int,
    base_channels: int,
    lr_factor: float,
    lr_patience: int,
    min_lr: float,
    checkpoint_dir: str,
    resume: str,
    save_every_epochs: int,
    export_ts: bool,
    device: str,
    seed: int,
    amp: bool,
    potato: bool,
) -> None:
    """Main training loop."""
    # Fall back to .env when --data-root is not provided via flag or env var
    if not data_root.strip():
        env_values = dotenv_values(".env")
        data_root = env_values.get("TRAIN_DATA_ROOT", data_root)

    # Potato mode: disable progress bars, force INFO logs
    if potato:
        os.environ["LOG_LEVEL"] = "INFO"
        os.environ["STEMMY_DISABLE_PROGRESS"] = "1"
        os.environ["EVAL_PROGRESS"] = "0"
        setup_logging(level="INFO")
    else:
        setup_logging()

    progress_disabled = _progress_disabled()
    if progress_disabled:
        logger.info("Progress bars disabled (STEMMY_DISABLE_PROGRESS=1)")

    # --- Inline validation (replaces validate_args) ---
    if epochs <= 0:
        raise click.ClickException("--epochs must be > 0")
    if batch_size <= 0:
        raise click.ClickException("--batch-size must be > 0")
    if lr <= 0:
        raise click.ClickException("--lr must be > 0")
    if weight_decay < 0:
        raise click.ClickException("--weight-decay must be >= 0")
    if num_workers < 0:
        raise click.ClickException("--num-workers must be >= 0")
    if time_frames not in [256, 512]:
        raise click.ClickException("--time-frames must be 256 or 512 for this baseline.")
    if save_every_epochs <= 0:
        raise click.ClickException("--save-every-epochs must be > 0")
    if max_tracks < 0:
        raise click.ClickException("--max-tracks must be >= 0")
    if base_channels <= 0:
        raise click.ClickException("--base-channels must be > 0")
    if lr_factor <= 0.0 or lr_factor >= 1.0:
        raise click.ClickException("--lr-factor must be in (0, 1)")
    if lr_patience < 0:
        raise click.ClickException("--lr-patience must be >= 0")
    if min_lr <= 0:
        raise click.ClickException("--min-lr must be > 0")
    if not isinstance(data_root, str) or data_root.strip() == "":
        raise click.ClickException("--data-root must be a non-empty string")
    if not isinstance(checkpoint_dir, str) or checkpoint_dir.strip() == "":
        raise click.ClickException("--checkpoint-dir must be a non-empty string")

    data_root_p = Path(data_root.strip()).expanduser()
    if not data_root_p.exists():
        raise click.ClickException(f"--data-root not found: {data_root_p}")
    if not data_root_p.is_dir():
        raise click.ClickException(f"--data-root must be a directory: {data_root_p}")

    ckpt_dir_p = Path(checkpoint_dir.strip()).expanduser()
    ckpt_dir_p.mkdir(parents=True, exist_ok=True)
    if not ckpt_dir_p.is_dir():
        raise click.ClickException(f"--checkpoint-dir must be a directory: {ckpt_dir_p}")

    if resume.strip():
        resume_p = Path(resume.strip()).expanduser()
        if not resume_p.exists():
            raise click.ClickException(f"--resume checkpoint not found: {resume_p}")
        if resume_p.is_dir():
            raise click.ClickException(
                f"--resume must point to a .pth file, got directory: {resume_p}"
            )

    if not isinstance(seed, int):
        raise click.ClickException("--seed must be an int")
    if seed < -1:
        raise click.ClickException("--seed must be -1 (disabled) or >= 0")

    # NOTE: Could be worth extracting setup to its own function
    # Initialize summary_rows for Training Summary Table
    summary_rows: list[dict[str, Any]] = []
    with create_setup_progress("Setup", title_style=BOLD_PURPLE) as setup_progress:
        setup_task = setup_progress.add_task(
            "Setup", total=11, step="Validating Args...", step_style=LOSS_STYLE
        )
        setup_progress.update(setup_task, advance=1, step="Seeding RNG...", step_style=LOSS_STYLE)
        if progress_disabled:
            logger.info("Setup: seeding RNG")
        set_seed(seed)
        setup_progress.update(
            setup_task, advance=1, step="Picking device...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: picking device")
        dev = pick_device(device)
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
        crop_cfg = CropConfig(time_frames=time_frames)
        setup_progress.update(
            setup_task, advance=1, step="Loading datasets...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: loading datasets")
        _train_ds, _val_ds, train_loader, val_loader = build_dataloaders(
            data_root=data_root,
            max_tracks=max_tracks,
            waveform_norm=waveform_norm,
            spectrogram_norm=spectrogram_norm,
            batch_size=batch_size,
            num_workers=num_workers,
            stft_cfg=stft_cfg,
            crop_cfg=crop_cfg,
            device=dev,
        )
        setup_progress.update(
            setup_task, advance=1, step="Initializing model...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing model")
        model = UNet2D(stems=len(STEMS_4), base_channels=base_channels).to(dev)
        setup_progress.update(
            setup_task, advance=1, step="Initializing optimizer...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        setup_progress.update(
            setup_task, advance=1, step="Initializing scheduler...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
        setup_progress.update(
            setup_task, advance=1, step="Initializing amp...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: initializing amp")
        use_amp = bool(amp) and (dev.type == "cuda")
        scaler = _make_grad_scaler(dev, use_amp)

        start_epoch = 0
        global_step = 0

        setup_progress.update(
            setup_task, advance=1, step="Restoring checkpoint...", step_style=LOSS_STYLE
        )
        if progress_disabled:
            logger.info("Setup: restoring checkpoint (if requested)")
        if resume.strip():
            ckpt_path = resume.strip()
            start_epoch, global_step, extra = load_checkpoint(
                ckpt_path,
                model,
                optimizer,
                map_location=dev,
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
        config = build_run_config(
            data_root=data_root,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            base_channels=base_channels,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            min_lr=min_lr,
            max_tracks=max_tracks,
            waveform_norm=waveform_norm,
            spectrogram_norm=spectrogram_norm,
            seed=seed,
            amp=amp,
            device=dev,
            stft_cfg=stft_cfg,
            crop_cfg=crop_cfg,
        )
        if wandb.run is not None:
            wandb.config.update(config)

        ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        setup_progress.update(setup_task, advance=1, step="▁▁▁▁▁", step_style=BAR_FINISHED_STYLE)

    # NOTE: Start of training & validation loop
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, train_batches = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=dev,
            use_amp=use_amp,
            scaler=scaler,
            epoch_index=epoch,
        )
        global_step += train_batches

        val_loss, _val_batches = eval_one_epoch(model, val_loader, dev, epoch_index=epoch)

        scheduler.step(val_loss)

        dt = time.time() - t0
        lr_now = float(optimizer.param_groups[0].get("lr", lr))
        logger.info(
            "Epoch %d/%d - lr=%.6e train_loss=%.6f val_loss=%.6f - time=%.1fs",
            int(epoch + 1),
            int(epochs),
            float(lr_now),
            float(train_loss),
            float(val_loss),
            float(dt),
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": lr_now,
                    "time/epoch_time": dt,
                },
                step=epoch + 1,
            )

        do_save = ((epoch + 1) % save_every_epochs) == 0
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

            if export_ts:
                ts_path = ckpt_dir / f"unet_phase1_epoch{epoch + 1:03d}.pt"
                export_torchscript(str(ts_path), model)
                saved_ts = str(ts_path)
                logger.info("Exported TorchScript: %s", str(ts_path))

        summary_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "checkpoint_path": saved_ckpt,
                "torchscript_path": saved_ts,
            }
        )

    print_training_summary(summary_rows, include_torchscript=bool(export_ts))


if __name__ == "__main__":
    main()
