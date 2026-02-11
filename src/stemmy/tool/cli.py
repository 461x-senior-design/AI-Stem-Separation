#########################
# Changed by Ryan
# Reason:
# Ruff E402 requires imports at the top of the module (comments are fine, but no
# executable code before imports). This CLI also extends Austin's preview-only
# command into a functional entry point using canonical stems and the project's
# inference pipeline, and configures centralized logging once at process startup.
#
# What it does:
# - Adds imports required for inference, device validation, and logging.
# - Converts the single command into a Click group with a `separate` subcommand.
# - Updates preview output naming to canonical: <base>_<stem>.wav.
# - Adds --checkpoint/--device/--preview and runs .pth inference to export stems.
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


#########################
# Changed by Ryan
# Reason:
# The CLI is now a Click group entry point (not a single command function), and
# logging should be configured once at startup for consistent formatting and
# third-party suppression.
# What it does:
# Defines a Click group and calls setup_logging() when invoked.
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
def separate(
    input_file: str,
    output_dir: str,
    checkpoint: str,
    device: str,
    preview: bool,
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
        )
    except TypeError:
        cfg.device = resolved_device
        cfg.stems = stems
        cfg.export_files = True
        cfg.renorm_masks = True

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
