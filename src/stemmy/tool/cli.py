# src/stemmy/tool/cli.py
"""Command-line interface for running stem separation inference.

This module provides a Click-based CLI that can:
- Preview expected output filenames for a given input track.
- Run end-to-end separation using either:
  - A training checkpoint (.pth), which enables config-from-checkpoint alignment, or
  - A TorchScript model (.pt), which is a frozen inference artifact.

Outputs are written as <base>_<stem>.wav files under the chosen output directory.
"""

import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import click
import torch
from dotenv import dotenv_values
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree

from stemmy.constants import BOLD_GREEN, BOLD_RED, CYAN, STEMS_4
from stemmy.inference import (
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    load_torchscript_model,
    separate_audio_file,
)
from stemmy.logging_config import setup_logging
from stemmy.tool.eq_frames import update_constants_eq_frames
from stemmy.tool.fullsong_eval_masked import main as fullsong_eval_masked_main
from stemmy.train import main as train_command

CHKPT: str = str((Path(__file__).resolve().parent / "default.pth").resolve())
DIR: str = Path.cwd().name
STEMS: list[str] = ["drums-", "vocals-", "bass-", "other-"]
TRAIN_DATA_ROOT_KEY: str = "TRAIN_DATA_ROOT"
DATA_KEY: str = "DATA"
CKPT_DIR_KEY: str = "CKPT_DIR"
EVAL_DIR_KEY: str = "EVAL_DIR"


def _apply_potato_mode(potato: bool) -> None:
    """Set terminal-friendly logging/progress defaults for dev workflows."""
    if not potato:
        return
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["STEMMY_DISABLE_PROGRESS"] = "1"
    os.environ["EVAL_PROGRESS"] = "0"
    setup_logging(level="INFO")


def _fullsong_eval_default_env() -> dict[str, str]:
    """Return default environment values for dev full-song evaluation."""
    env_values = dotenv_values(".env")

    data_root = os.getenv(TRAIN_DATA_ROOT_KEY) or env_values.get(TRAIN_DATA_ROOT_KEY) or ""
    defaults: dict[str, str] = {}

    if isinstance(data_root, str) and data_root.strip():
        defaults[DATA_KEY] = data_root.strip()

    defaults[CKPT_DIR_KEY] = "checkpoints"
    defaults[EVAL_DIR_KEY] = "eval"
    return defaults


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

    resolved_device = _resolve_device(device)

    if ckpt_in != "":
        ckpt_path = Path(ckpt_in).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError("Checkpoint not found: %s" % str(ckpt_path))
        if not ckpt_path.is_file():
            raise IsADirectoryError(
                "Expected a checkpoint file, got a directory: %s" % str(ckpt_path)
            )

        model, ckpt_obj = load_pth_model(str(ckpt_path), device=resolved_device, stems=len(stems))
        cfg = config_from_checkpoint(ckpt_obj)
        return model, cfg, ckpt_obj

    ts_path = Path(ts_in).expanduser().resolve()
    if not ts_path.exists():
        raise FileNotFoundError("TorchScript model not found: %s" % str(ts_path))
    if not ts_path.is_file():
        raise IsADirectoryError("Expected a TorchScript file, got a directory: %s" % str(ts_path))

    model = load_torchscript_model(str(ts_path), device=resolved_device)
    cfg = InferenceConfig()
    return model, cfg, None


# NOTE: Root CLI command group
@click.group()
def cli() -> None:
    """Stemmy command-line interface."""
    setup_logging(level=os.getenv("LOG_LEVEL", "ERROR"))


# NOTE: `stemmy dev` command group
@cli.group(help="Developer commands.")
def dev() -> None:
    """Developer command group."""


# NOTE: `stemmy dev train` — registered from stemmy.train as a first-class Click command
dev.add_command(train_command)


# NOTE: `stemmy dev eval` command
@dev.command(
    "eval",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help=(
        "Run full-song masked evaluation. "
        "Pass environment overrides as KEY=VALUE arguments "
        "(for example DATA=/path CKPT_DIR=/path EVAL_DIR=/path)."
    ),
)
@click.option(
    "--potato",
    is_flag=True,
    default=False,
    help="Disable progress bars and force INFO logs.",
)
@click.argument("env_args", nargs=-1, type=click.UNPROCESSED)
def dev_fullsong_eval_masked(potato: bool, env_args: Sequence[str]) -> None:
    """Run full-song evaluation via stemmy.tool.fullsong_eval_masked."""
    env_overrides = [arg for arg in env_args if arg != "--potato"]
    potato = bool(potato or ("--potato" in env_args))
    _apply_potato_mode(potato)
    os.environ.setdefault("LOG_LEVEL", "ERROR")

    for key, value in _fullsong_eval_default_env().items():
        os.environ.setdefault(key, value)

    for raw in env_overrides:
        if "=" not in raw:
            raise click.ClickException(
                "Invalid argument '%s'. Expected KEY=VALUE environment override." % raw
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        if key == "":
            raise click.ClickException("Invalid argument '%s'. Empty environment key." % raw)
        os.environ[key] = value

    fullsong_eval_masked_main()


# NOTE: `stemmy spinner` command
@cli.command("spinner", help="Generate EQ spinner frames and update src/stemmy/constants.py.")
def spinner() -> None:
    """Regenerate EQ_FRAMES and write them to constants.py."""
    try:
        changed = update_constants_eq_frames()
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        raise click.ClickException("Failed to update EQ_FRAMES: %s" % exc) from exc

    if changed:
        click.echo("Updated EQ_FRAMES in src/stemmy/constants.py")
    else:
        click.echo("EQ_FRAMES already up to date in src/stemmy/constants.py")


# NOTE: `stemmy gui` command
@cli.command("gui", help="Launch the Stemmy graphical front end.")
def gui() -> None:
    """Launch the PySide6 GUI."""
    try:
        from stemmy.gui.gui import main as gui_main
    except ImportError as exc:
        raise click.ClickException(
            "Unable to import the GUI. Install the GUI dependency with: "
            'pip install -e ".[gui]"'
        ) from exc

    raise SystemExit(gui_main())


# NOTE: `stemmy separate` command
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
@click.option("--output-dir", "-o", default=".", help="Name of output directory.")
@click.option(
    "--checkpoint",
    "-c",
    default=CHKPT,
    show_default=True,
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
    default=256,  # NOTE: Changed from 0
    show_default=True,
    help=(
        "If > 0, run inference in time chunks of this many STFT frames to "
        "reduce VRAM. 0 disables chunking."
    ),
)
@click.option(
    "--overlap-frames",
    type=int,
    default=64,  # NOTE: Changed from 0
    show_default=True,
    help="Overlap (in STFT frames) between chunks when --chunk-frames > 0.",
)
@click.option(
    "--amp/--no-amp",
    default=False,
    show_default=True,
    help="Enable CUDA autocast (AMP) during inference to reduce memory usage.",
)
@click.option(
    "--potato",
    is_flag=True,
    default=False,
    help="Disable progress bars and force INFO logs.",
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
    potato: bool,
) -> None:
    """Separate an input file into stems and export the outputs."""
    _apply_potato_mode(potato)

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

    model, cfg, ckpt_obj = _load_model_and_cfg(
        checkpoint=checkpoint,
        torchscript=torchscript,
        device=device,
        stems=stems,
    )

    resolved_device = _resolve_device(device)

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
