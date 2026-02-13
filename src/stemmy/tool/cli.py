# src/stemmy/tool/cli.py
"""Command-line interface for running stem separation inference.

This module provides a Click-based CLI that can:
- Preview expected output filenames for a given input track.
- Run end-to-end separation using either:
  - A training checkpoint (.pth), which enables config-from-checkpoint alignment, or
  - A TorchScript model (.pt), which is a frozen inference artifact.

Outputs are written as <base>_<stem>.wav files under the chosen output directory.
"""

from dataclasses import replace
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree

from stemmy.constants import STEMS_4
from stemmy.inference import (
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    load_torchscript_model,
    separate_audio_file,
)
from stemmy.logging_config import setup_logging

BOLD_GREEN: str = "bold green"
BOLD_RED: str = "bold red"
CYAN: str = "cyan"
DIR: str = Path.cwd().name
STEMS: list[str] = ["drums-", "vocals-", "bass-", "other-"]


def _resolve_device(device: str) -> str:
    """Validate and normalize a device string into one of: cpu | cuda | cuda:N."""
    dev_in = (device or "cpu").strip()
    if dev_in == "":
        dev_in = "cpu"

    if dev_in == "cpu":
        return "cpu"

    if dev_in == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"

    if dev_in.startswith("cuda:"):
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
        return "cuda:%d" % idx

    raise ValueError("Invalid --device (must be cpu|cuda|cuda:N): %s" % dev_in)


def _validate_chunking(chunk_frames: int, overlap_frames: int) -> None:
    """Validate chunking arguments used to reduce VRAM via time-chunked inference."""
    if not isinstance(chunk_frames, int) or chunk_frames < 0:
        raise click.ClickException("--chunk-frames must be >= 0.")
    if not isinstance(overlap_frames, int) or overlap_frames < 0:
        raise click.ClickException("--overlap-frames must be >= 0.")
    if chunk_frames > 0 and overlap_frames >= chunk_frames:
        raise click.ClickException(
            "--overlap-frames must be < --chunk-frames when chunking is enabled."
        )


def _validate_input_file(path_str: str) -> Path:
    """Validate and return the resolved input audio path."""
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError("Input file not found: %s" % str(p))
    if not p.is_file():
        raise IsADirectoryError("Expected a file, got a directory: %s" % str(p))
    return p


def _prepare_output_dir(path_str: str) -> Path:
    """Ensure output directory exists and return the resolved path."""
    p = Path(path_str).expanduser().resolve()
    if p.exists() and not p.is_dir():
        raise NotADirectoryError("Output path exists but is not a directory: %s" % str(p))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_model_and_cfg(
    checkpoint: str,
    torchscript: str,
    device: str,
    stems: list[str],
) -> tuple[torch.nn.Module, InferenceConfig, object]:
    """Load a model and construct an InferenceConfig.

    Args:
        checkpoint: Path to a .pth checkpoint (optional).
        torchscript: Path to a .pt TorchScript model (optional).
        device: A validated/normalized device string: cpu | cuda | cuda:N.
        stems: List of stem names (used to size certain model heads for .pth loading).

    Returns:
        (model, cfg, checkpoint_obj)

    Notes:
        - For .pth: loads PyTorch model weights and derives cfg from checkpoint metadata.
        - For .pt: loads TorchScript and uses default InferenceConfig (no checkpoint metadata).
    """
    ckpt_in = checkpoint.strip() if isinstance(checkpoint, str) and checkpoint is not None else ""
    ts_in = torchscript.strip() if isinstance(torchscript, str) and torchscript is not None else ""

    if ckpt_in == "" and ts_in == "":
        raise click.ClickException(
            "Either --checkpoint (.pth) or --torchscript (.pt) is required unless --preview is set."
        )
    if ckpt_in != "" and ts_in != "":
        raise click.ClickException("Use only one: --checkpoint or --torchscript (not both).")

    if ckpt_in != "":
        ckpt_path = Path(ckpt_in).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError("Checkpoint not found: %s" % str(ckpt_path))
        if not ckpt_path.is_file():
            raise IsADirectoryError(
                "Expected a checkpoint file, got a directory: %s" % str(ckpt_path)
            )

        model, ckpt_obj = load_pth_model(str(ckpt_path), device=device, stems=len(stems))
        cfg = config_from_checkpoint(ckpt_obj)
        return model, cfg, ckpt_obj

    ts_path = Path(ts_in).expanduser().resolve()
    if not ts_path.exists():
        raise FileNotFoundError("TorchScript model not found: %s" % str(ts_path))
    if not ts_path.is_file():
        raise IsADirectoryError("Expected a TorchScript file, got a directory: %s" % str(ts_path))

    model = load_torchscript_model(str(ts_path), device=device)
    cfg = InferenceConfig()
    return model, cfg, None


@click.group()
def cli() -> None:
    """Stemmy command-line interface."""
    setup_logging()


@cli.command(
    help=(
        "Separate an audio file into stems.\n\n"
        "Choose exactly one model artifact:\n"
        "  -c/--checkpoint (.pth): preferred when you want config derived from training metadata.\n"
        "  -t/--torchscript (.pt): preferred when you want a frozen inference artifact.\n\n"
        "Outputs:\n"
        "  Writes <base>_<stem>.wav under -o/--output-dir.\n\n"
        "Use --preview to print expected output names without running inference."
    )
)
@click.option("--input-file", "-i", required=True, help="Name of input audio file.")
@click.option("--output-dir", "-o", default=DIR, help="Name of output directory.")
@click.option(
    "--checkpoint",
    "-c",
    required=False,
    help=(
        "Path to model checkpoint (.pth). Preferred when you want config-from-checkpoint "
        "alignment. Mutually exclusive with --torchscript."
    ),
)
@click.option(
    "--torchscript",
    "-t",
    required=False,
    help=(
        "Path to TorchScript model (.pt). Preferred when you want a frozen inference artifact. "
        "Mutually exclusive with --checkpoint."
    ),
)
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
    torchscript: str,
    device: str,
    preview: bool,
    chunk_frames: int,
    overlap_frames: int,
    amp: bool,
) -> None:
    """Separate an input file into stems and export the outputs."""
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

    _validate_chunking(chunk_frames=int(chunk_frames), overlap_frames=int(overlap_frames))

    input_path = _validate_input_file(input_file)
    out_dir_path = _prepare_output_dir(output_dir)

    resolved_device = _resolve_device(device)

    model, cfg, ckpt_obj = _load_model_and_cfg(
        checkpoint=checkpoint,
        torchscript=torchscript,
        device=resolved_device,
        stems=stems,
    )

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
