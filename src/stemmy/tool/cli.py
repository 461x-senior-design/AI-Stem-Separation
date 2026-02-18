from dataclasses import replace
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree

from stemmy.constants import CLI_COLOR_ERROR, CLI_COLOR_INFO, CLI_COLOR_SUCCESS, STEMS_4
from stemmy.inference import config_from_checkpoint, load_pth_model, separate_audio_file

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
    pass


#########################


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
    console.print("\nSong Name:", style=CLI_COLOR_ERROR, end=" ")
    console.print(display_input, style=CLI_COLOR_INFO)
    console.print("\nExpected Output:", style=CLI_COLOR_ERROR)

    tree = Tree(Text(output_dir, style=CLI_COLOR_SUCCESS))

    #########################
    # Changed by Ryan
    # Reason:
    #   Display uses canonical stems and naming.
    #   Canonical .pth inference integration.

    stems = list(STEMS_4)
    base = Path(display_input).stem
    for stem in stems:
        tree.add(Text(f"{base}_{stem}.wav", style=CLI_COLOR_INFO))

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
    #########################


if __name__ == "__main__":
    cli()
