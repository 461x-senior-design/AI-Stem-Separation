# src/stemmy/tool/cli.py
#########################
# Changed by Ryan
# Reason:
# Extend Austin's preview-only CLI into a functional entry point that runs the
# canonical inference pipeline, with explicit device validation and optional
# low-VRAM inference controls (chunked inference and AMP).
#
# What it does:
# - Adds the missing imports for checkpoint loading, inference, and CUDA checks.
# - Keeps Austin's original UI/printing lines unchanged where possible.
from dataclasses import replace
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree

from stemmy.constants import STEMS_4
from stemmy.inference import config_from_checkpoint, load_pth_model, separate_audio_file
from stemmy.logging_config import setup_logging

#########################

BOLD_GREEN: str = "bold green"
BOLD_RED: str = "bold red"
CYAN: str = "cyan"
DIR: str = Path.cwd().name
STEMS: list[str] = ["drums-", "vocals-", "bass-", "other-"]


#########################
# Changed by Ryan
# Reason:
# Convert the single-command stub into a Click group with a real `separate`
# subcommand that:
# - Preserves Austin's preview output behavior.
# - Adds --checkpoint/--device for actual inference.
# - Adds --chunk-frames/--overlap-frames/--amp to reduce VRAM usage on GPU.
# - Uses canonical stem naming: <base>_<stem>.wav (based on STEMS_4 ordering).
@click.group()
def cli() -> None:
    """Stemmy command-line interface."""
    setup_logging()


@cli.command()
@click.option("--input-file", "-i", required=True, help="Name of input audio file.")
@click.option("--output-dir", "-o", default=DIR, help="Name of output directory.")
@click.option("--checkpoint", "-c", required=False, help="Path to model checkpoint (.pth).")
@click.option(
    "--device",
    "-d",
    default="cpu",
    show_default=True,
    help="Device for inference: cpu | cuda | cuda:N",
)
@click.option(
    "--preview",
    is_flag=True,
    default=False,
    help="Print expected output names and exit without running inference.",
)
@click.option(
    "--chunk-frames",
    type=int,
    default=0,
    show_default=True,
    help=(
        "If > 0, run inference in time chunks of this many STFT frames to "
        "reduce VRAM. 0 disables chunking."
    ),
)
@click.option(
    "--overlap-frames",
    type=int,
    default=0,
    show_default=True,
    help="Overlap (in STFT frames) between chunks when --chunk-frames > 0.",
)
@click.option(
    "--amp/--no-amp",
    default=False,
    show_default=True,
    help="Enable CUDA autocast (AMP) during inference to reduce memory usage.",
)
def separate(
    input_file: str,
    output_dir: str,
    checkpoint: str,
    device: str,
    preview: bool,
    chunk_frames: int,
    overlap_frames: int,
    amp: bool,
) -> None:
    """CLI wrapper for the separate command."""
    console = Console()
    display_input = input_file
    console.print("\nSong Name:", style=BOLD_RED, end=" ")
    console.print(display_input, style=CYAN)
    console.print("\nExpected Output:", style=BOLD_RED)
    tree = Tree(Text(output_dir, style=BOLD_GREEN))

    stems = list(STEMS_4)
    base = Path(display_input).stem
    for stem in stems:
        tree.add(Text(f"{base}_{stem}.wav", style=CYAN))

    console.print(Padding(tree, (0, 0, 0, 2)))

    if preview:
        return

    if checkpoint is None or checkpoint.strip() == "":
        raise click.ClickException("--checkpoint is required unless --preview is set.")

    if not isinstance(chunk_frames, int) or chunk_frames < 0:
        raise click.ClickException("--chunk-frames must be >= 0.")
    if not isinstance(overlap_frames, int) or overlap_frames < 0:
        raise click.ClickException("--overlap-frames must be >= 0.")
    if chunk_frames > 0 and overlap_frames >= chunk_frames:
        raise click.ClickException(
            "--overlap-frames must be < --chunk-frames when chunking is enabled."
        )

    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError("Input file not found: %s" % str(input_path))
    if not input_path.is_file():
        raise IsADirectoryError("Expected a file, got a directory: %s" % str(input_path))

    out_dir_path = Path(output_dir).expanduser().resolve()
    if out_dir_path.exists() and not out_dir_path.is_dir():
        raise NotADirectoryError(
            "Output path exists but is not a directory: %s" % str(out_dir_path)
        )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found: %s" % str(ckpt_path))
    if not ckpt_path.is_file():
        raise IsADirectoryError("Expected a checkpoint file, got a directory: %s" % str(ckpt_path))

    dev_in = (device or "cpu").strip()
    if dev_in == "":
        dev_in = "cpu"

    if dev_in == "cpu":
        resolved_device = "cpu"
    elif dev_in == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        resolved_device = "cuda"
    elif dev_in.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        idx_str = dev_in.split(":", 1)[1].strip()
        if idx_str == "":
            raise ValueError("Invalid --device value (expected cuda:N): %s" % dev_in)
        try:
            idx = int(idx_str)
        except ValueError as exc:
            raise ValueError("Invalid --device value (expected cuda:N): %s" % dev_in) from exc
        if idx < 0:
            raise ValueError("Invalid --device GPU index: %d" % idx)
        resolved_device = "cuda:%d" % idx
    else:
        raise ValueError("Invalid --device (must be cpu|cuda|cuda:N): %s" % dev_in)

    model, ckpt_obj = load_pth_model(str(ckpt_path), device=resolved_device, stems=len(stems))
    cfg = config_from_checkpoint(ckpt_obj)

    try:
        cfg = replace(
            cfg,
            device=resolved_device,
            stems=stems,
            export_files=True,
            renorm_masks=True,
            chunk_frames=int(chunk_frames),
            overlap_frames=int(overlap_frames),
            amp=bool(amp),
        )
    except TypeError:
        cfg.device = resolved_device
        cfg.stems = stems
        cfg.export_files = True
        cfg.renorm_masks = True
        cfg.chunk_frames = int(chunk_frames)
        cfg.overlap_frames = int(overlap_frames)
        cfg.amp = bool(amp)

    separate_audio_file(
        audio_path=input_path,
        model=model,
        cfg=cfg,
        output_dir=out_dir_path,
        export_files=True,
        stems=stems,
        checkpoint=ckpt_obj,
    )


if __name__ == "__main__":
    cli()
#########################
