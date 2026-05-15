"""Shared Click option groups for the stemmy CLI.

Each list contains related ``click.option`` decorators that can be applied
to a command via the ``add_options`` helper, keeping command definitions concise.
"""

import click

from stemmy.constants import DEFAULT_SPECTROGRAM_NORM, DEFAULT_WAVEFORM_NORM


def add_options(options):
    """Decorator that applies a list of click.option decorators to a function."""

    def wrapper(func):
        for opt in reversed(options):
            func = opt(func)
        return func

    return wrapper


processing_options = [
    click.option(
        "--data-root",
        type=str,
        default="",
        envvar=["TRAIN_DATA_ROOT", "DATA_ROOT"],
        show_default=False,
        help="Root directory containing the dataset splits.",
    ),
    click.option("--batch-size", type=int, default=4, show_default=True, help="Mini-batch size."),
    click.option(
        "--num-workers", type=int, default=2, show_default=True, help="DataLoader worker processes."
    ),
    click.option(
        "--max-tracks",
        type=int,
        default=0,
        show_default=True,
        help="If > 0, limit number of tracks per split.",
    ),
    click.option(
        "--waveform-norm",
        type=click.Choice(["peak", "rms", "none"]),
        default=DEFAULT_WAVEFORM_NORM,
        show_default=True,
        help="Waveform normalization method.",
    ),
    click.option(
        "--spectrogram-norm",
        type=click.Choice(["freq_minmax", "none"]),
        default=DEFAULT_SPECTROGRAM_NORM,
        show_default=True,
        help="Spectrogram normalization method.",
    ),
    click.option(
        "--time-frames",
        type=int,
        default=256,
        show_default=True,
        help="Fixed STFT time frames T (256 or 512).",
    ),
]

training_options = [
    click.option(
        "--epochs", type=int, default=1, show_default=True, help="Number of training epochs."
    ),
    click.option(
        "--base-channels",
        type=int,
        default=64,
        show_default=True,
        help="Base channel count for the U-Net.",
    ),
    click.option("--lr", type=float, default=1e-4, show_default=True, help="Learning rate."),
    click.option(
        "--weight-decay",
        type=float,
        default=0.0,
        show_default=True,
        help="Weight decay for AdamW (>= 0).",
    ),
    click.option(
        "--lr-factor",
        type=float,
        default=0.5,
        show_default=True,
        help="ReduceLROnPlateau factor in (0,1).",
    ),
    click.option(
        "--lr-patience",
        type=int,
        default=10,
        show_default=True,
        help="ReduceLROnPlateau patience (epochs).",
    ),
    click.option(
        "--min-lr",
        type=float,
        default=1e-6,
        show_default=True,
        help="ReduceLROnPlateau minimum LR.",
    ),
    click.option(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        show_default=True,
        help="Output directory for checkpoints.",
    ),
    click.option(
        "--resume",
        type=str,
        default="",
        show_default=False,
        help="Path to a .pth checkpoint to resume from.",
    ),
    click.option(
        "--save-every-epochs",
        type=int,
        default=1,
        show_default=True,
        help="Save a checkpoint every N epochs.",
    ),
    click.option(
        "--export-ts",
        is_flag=True,
        default=False,
        help="Export TorchScript .pt after each checkpoint save.",
    ),
    click.option(
        "--device",
        type=str,
        default="",
        show_default=False,
        help="cpu, cuda, or leave empty for auto-selection.",
    ),
    click.option(
        "--seed",
        type=int,
        default=-1,
        show_default=True,
        help="If >= 0, seed RNGs for reproducibility.",
    ),
    click.option("--amp", is_flag=True, default=False, help="Enable AMP (CUDA only)."),
    click.option(
        "--potato", is_flag=True, default=False, help="Disable progress bars and force INFO logs."
    ),
]

augmentation_options = [
    click.option(
        "--aug-gain-p",
        type=float,
        default=0.0,
        envvar="AUG_GAIN_P",
        show_default=True,
        help="Probability of applying random gain. 0 disables.",
    ),
    click.option(
        "--aug-gain-db",
        type=float,
        default=6.0,
        envvar="AUG_GAIN_DB",
        show_default=True,
        help="Max |gain| in dB; sampled uniformly from [-X, +X].",
    ),
    click.option(
        "--aug-pitch-p",
        type=float,
        default=0.0,
        envvar="AUG_PITCH_P",
        show_default=True,
        help="Probability of applying pitch shift. 0 disables.",
    ),
    click.option(
        "--aug-pitch-semitones",
        type=float,
        default=2.0,
        envvar="AUG_PITCH_SEMITONES",
        show_default=True,
        help="Max |pitch shift| in semitones; sampled uniformly from [-X, +X].",
    ),
    click.option(
        "--aug-time-stretch-p",
        type=float,
        default=0.0,
        envvar="AUG_TIME_STRETCH_P",
        show_default=True,
        help="Probability of applying time stretch. 0 disables.",
    ),
    click.option(
        "--aug-time-stretch-range",
        type=float,
        default=0.1,
        envvar="AUG_TIME_STRETCH_RANGE",
        show_default=True,
        help="Time-stretch rate range; sampled uniformly from [1-X, 1+X]. Length preserved.",
    ),
    click.option(
        "--aug-shift-p",
        type=float,
        default=0.0,
        envvar="AUG_SHIFT_P",
        show_default=True,
        help="Probability of applying random time shift. 0 disables.",
    ),
    click.option(
        "--aug-shift-fraction",
        type=float,
        default=0.1,
        envvar="AUG_SHIFT_FRACTION",
        show_default=True,
        help="Max |shift| as fraction of segment; sampled uniformly from [-X, +X]. Rolls over.",
    ),
    click.option(
        "--aug-polarity-p",
        type=float,
        default=0.0,
        envvar="AUG_POLARITY_P",
        show_default=True,
        help="Probability of applying polarity inversion. 0 disables.",
    ),
    click.option(
        "--aug-noise-p",
        type=float,
        default=0.0,
        envvar="AUG_NOISE_P",
        show_default=True,
        help="Probability of adding Gaussian noise. 0 disables.",
    ),
    click.option(
        "--aug-noise-amplitude",
        type=float,
        default=0.005,
        envvar="AUG_NOISE_AMPLITUDE",
        show_default=True,
        help="Max amplitude of additive Gaussian noise; sampled uniformly from [0, X].",
    ),
]
