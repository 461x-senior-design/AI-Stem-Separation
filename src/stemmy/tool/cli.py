#########################
# cli.py
# CLI entry point for Stemmy.

from pathlib import Path

import click
import torch
from rich.console import Console

from src.constants import STEMS_4
from src.inference import config_from_checkpoint, load_pth_model, separate_audio_file
from src.logging_config import get_logger

logger = get_logger(__name__)


class CliException(Exception):
    """Raised for predictable CLI failures."""


#########################
# Change by Ryan:
# Reason:
# Validate device strings (cpu/cuda/cuda:N) and refuse CUDA when unavailable.
def _validate_device(device: str) -> str:
    dev = (device or "").strip()
    if dev == "":
        dev = "cpu"

    if dev == "cpu":
        return dev

    if dev == "cuda" or dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise CliException("Requested CUDA device but torch.cuda.is_available() is False.")
        return dev

    raise CliException("Invalid --device. Use cpu, cuda, or cuda:N (example: cuda:0).")
#########################


#########################
# Change by Ryan:
# Reason:
# Create timestamped run directories under the base output directory and
# write a command.txt file for later comparisons.
def _timestamp_tag() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_run_dir(base_output_dir: Path) -> Path:
    tag = _timestamp_tag()
    run_dir = base_output_dir / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_command_file(run_dir: Path) -> None:
    import os
    import shlex
    import sys

    cmd = " ".join(shlex.quote(a) for a in sys.argv)
    cwd = os.getcwd()

    (run_dir / "command.txt").write_text(
        "cwd: %s\ncmd: %s\n" % (cwd, cmd),
        encoding="utf-8",
    )
#########################


@click.group()
def cli() -> None:
    """Stemmy command-line interface."""


@cli.command("separate")
@click.option("--input-file", "-i", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--checkpoint", "-c", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir",
    "-o",
    default="runs/separated_cli",
    type=click.Path(file_okay=False),
    help="Base output directory. A timestamped subdirectory is created inside it.",
)
@click.option("--device", default="cpu", help="cpu, cuda, or cuda:N")
def separate(input_file: str, checkpoint: str, output_dir: str, device: str) -> None:
    """Separate a single audio file into stems."""
    console = Console()

    try:
        dev = _validate_device(device)
    except CliException as exc:
        raise click.ClickException(str(exc))

    base_out_dir = Path(output_dir).expanduser().resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    #########################
    # Change by Ryan:
    # Reason:
    # Always use a timestamped output run folder and log the exact command.
    run_dir = _make_run_dir(base_out_dir)
    _write_command_file(run_dir)
    #########################

    ckpt_path = Path(checkpoint).expanduser().resolve()
    in_path = Path(input_file).expanduser().resolve()

    logger.info(
        "CLI separate: input=%s checkpoint=%s base_output_dir=%s run_dir=%s device=%s",
        str(in_path),
        str(ckpt_path),
        str(base_out_dir),
        str(run_dir),
        dev,
    )

    console.print(f"[bold green]Run directory[/bold green] {run_dir}")
    console.print(f"[bold green]Loading model[/bold green] from {ckpt_path}")

    model, ckpt = load_pth_model(str(ckpt_path), device=dev, stems=len(STEMS_4))
    cfg = config_from_checkpoint(ckpt)

    cfg.device = dev
    cfg.export_files = True
    cfg.renorm_masks = True
    cfg.stems = list(STEMS_4)

    outputs = separate_audio_file(
        audio_path=in_path,
        model=model,
        cfg=cfg,
        output_dir=run_dir,
        export_files=True,
        stems=list(STEMS_4),
        checkpoint=ckpt,
    )

    console.print("[bold green]Exported stems:[/bold green]")
    for name in STEMS_4:
        console.print(f"  {name}: {outputs.stem_paths.get(name)}")


if __name__ == "__main__":
    cli()

