import time
from pathlib import Path

import click
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.tree import Tree

BOLD_GREEN: str = "bold green"
BOLD_RED: str = "bold red"
CYAN: str = "cyan"
DIR: str = Path.cwd().name
STEMS: list[str] = ["drums-", "vocals-", "bass-", "other-"]


#########################
# Change by Ryan:
# Reason:
# The original teammate stub only prints expected output. End-to-end separation requires calling the
# unified inference stack with a checkpoint and (optionally) a device selection, without changing
# the original stub text.
# What it does:
# Re-defines the Click command named "separate" with additional options (-c/--checkpoint, --device).
# Validates inputs, loads the model/checkpoint using the project's unified inference utilities.
# Builds an InferenceConfig from checkpoint metadata, and runs separation to export stems into the
# output directory. The new command replaces the stub by re-binding the name `separate` before the
# __main__ call runs.
#########################


def _import_inference_stack():
    try:
        from stemmy.constants import STEMS_4
        from stemmy.inference import (
            InferenceConfig,
            config_from_checkpoint,
            load_pth_model,
            separate_audio_file,
        )

        return STEMS_4, InferenceConfig, config_from_checkpoint, load_pth_model, separate_audio_file
    except Exception:
        pass

    try:
        from src.stemmy.constants import STEMS_4
        from src.stemmy.inference import (
            InferenceConfig,
            config_from_checkpoint,
            load_pth_model,
            separate_audio_file,
        )

        return STEMS_4, InferenceConfig, config_from_checkpoint, load_pth_model, separate_audio_file
    except Exception as e:
        raise click.ClickException(
            (
                "Failed to import inference stack. Neither 'stemmy.*' nor 'src.stemmy.*' "
                "imports worked. "
                f"Import error: {e}"
            )
        )


def _get_attr_or_key(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _validate_paths(input_file: str, output_dir: str, checkpoint: str):
    in_path = Path(input_file).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    ckpt_path = Path(checkpoint).expanduser().resolve()

    if not in_path.exists():
        raise click.ClickException(f"Input file does not exist: {in_path}")
    if not in_path.is_file():
        raise click.ClickException(f"Input path is not a file: {in_path}")

    if not ckpt_path.exists():
        raise click.ClickException(f"Checkpoint does not exist: {ckpt_path}")
    if not ckpt_path.is_file():
        raise click.ClickException(f"Checkpoint path is not a file: {ckpt_path}")

    if out_dir.exists() and not out_dir.is_dir():
        raise click.ClickException(f"Output path exists but is not a directory: {out_dir}")

    return in_path, out_dir, ckpt_path


def _validate_device(device: str) -> str:
    dev = (device or "").strip()
    if dev == "":
        dev = "cpu"

    if dev == "cpu":
        return dev

    if dev == "cuda" or dev.startswith("cuda:"):
        try:
            import torch
        except Exception as e:
            raise click.ClickException(f"CUDA device requested but torch import failed: {e}")

        if not torch.cuda.is_available():
            raise click.ClickException(
                "Requested CUDA device but torch.cuda.is_available() is False."
            )
        return dev

    raise click.ClickException('Invalid --device. Expected "cpu", "cuda", or "cuda:N".')


@click.command(name="separate")
@click.option("--input-file", "-i", required=True, type=str, help="Path to input audio file.")
@click.option("--output-dir", "-o", default=DIR, type=str, help="Output directory.")
@click.option("--checkpoint", "-c", required=True, type=str, help="Path to .pth checkpoint file.")
@click.option("--device", default="cpu", type=str, help='Device: "cpu", "cuda", or "cuda:N".')
def separate(input_file: str, output_dir: str, checkpoint: str, device: str) -> None:
    """CLI wrapper for the separate command."""
    # Setup logging for CLI usage
    try:
        from stemmy.logging_config import setup_logging
        setup_logging(level="INFO")
    except ImportError:
        pass

    console = Console()
    t_start = time.perf_counter()

    in_path, out_dir, ckpt_path = _validate_paths(input_file, output_dir, checkpoint)
    dev = _validate_device(device)

    out_dir.mkdir(parents=True, exist_ok=True)

    STEMS_4, InferenceConfig, config_from_checkpoint, load_pth_model, separate_audio_file = (
        _import_inference_stack()
    )

    console.print("\nSong Name:", style=BOLD_RED, end=" ")
    console.print(str(in_path), style=CYAN)
    console.print("\nExpected Output:", style=BOLD_RED)
    tree = Tree(Text(str(out_dir), style=BOLD_GREEN))
    for stem in STEMS:
        tree.add(Text(f"{stem}{in_path.name}", style=CYAN))
    console.print(Padding(tree, (0, 0, 0, 2)))

    try:
        model, ckpt_obj = load_pth_model(str(ckpt_path), device=dev, stems=len(STEMS_4))
    except TypeError:
        model, ckpt_obj = load_pth_model(str(ckpt_path), device=dev, stems=list(STEMS_4))

    base_cfg_raw = config_from_checkpoint(ckpt_obj)

    sample_rate = int(_get_attr_or_key(base_cfg_raw, "sample_rate", 44100))
    n_fft = int(_get_attr_or_key(base_cfg_raw, "n_fft", 4096))
    hop_length = int(_get_attr_or_key(base_cfg_raw, "hop_length", 1024))
    win_length = int(_get_attr_or_key(base_cfg_raw, "win_length", n_fft))
    window = str(_get_attr_or_key(base_cfg_raw, "window", "hann"))
    center = bool(_get_attr_or_key(base_cfg_raw, "center", True))
    waveform_norm = str(_get_attr_or_key(base_cfg_raw, "waveform_norm", "peak"))
    spectrogram_norm = str(_get_attr_or_key(base_cfg_raw, "spectrogram_norm", "minmax"))
    eps = float(_get_attr_or_key(base_cfg_raw, "eps", 1e-8))
    validate_outputs = bool(_get_attr_or_key(base_cfg_raw, "validate_outputs", False))

    try:
        cfg = InferenceConfig(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            waveform_norm=waveform_norm,
            spectrogram_norm=spectrogram_norm,
            eps=eps,
            device=dev,
            stems=list(STEMS_4),
            export_files=True,
            renorm_masks=True,
            validate_outputs=validate_outputs,
        )
    except TypeError as e:
        raise click.ClickException(f"InferenceConfig constructor mismatch. Error: {e}")

    try:
        separate_audio_file(
            audio_path=in_path,
            model=model,
            cfg=cfg,
            output_dir=out_dir,
            export_files=True,
            stems=list(STEMS_4),
            checkpoint=ckpt_obj,
        )
    except TypeError:
        separate_audio_file(
            audio_path=in_path,
            model=model,
            cfg=cfg,
            output_dir=out_dir,
            export_files=True,
            stems=list(STEMS_4),
        )

    dt = time.perf_counter() - t_start
    console.print(f"\nDone in {dt:.1f}s.", style=BOLD_GREEN)


if __name__ == "__main__":
    separate()
