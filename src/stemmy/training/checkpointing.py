# src/stemmy/training/checkpointing.py
"""Checkpoint IO helpers for training and inference.

This module defines a small, backwards-compatible checkpoint format:

- payload["model_state"]: model.state_dict()
- payload["optimizer_state"]: optimizer.state_dict() (optional on load)
- payload["epoch"]: int
- payload["step"]: int
- payload["extra"]: dict (optional)

The extra dict may include:
- extra["config"]: a minimal end-to-end config dict needed to keep preprocessing,
  training, and inference consistent (STFT framing, stem ordering, normalization).
- extra["model"]: optional model metadata.

The validators in this file are intentionally conservative:
they enforce basic types for fields that affect end-to-end consistency, while
remaining compatible with older checkpoints that omit fields.
"""

from pathlib import Path
from typing import Optional, Union

import torch

from stemmy.logging_config import get_logger

logger = get_logger(__name__)


class CheckpointFormatException(Exception):
    """Raised when a checkpoint payload is invalid or incompatible with this code."""

    pass


def _validate_config_dict(config: dict) -> None:
    """Validate a checkpoint config dict.

    This is intentionally minimal to preserve backward compatibility.

    The goal is to prevent subtle end-to-end drift. The fields checked here are
    the ones that can silently break consistency between training and inference
    if they change types or shapes.

    Args:
        config: A dict expected to contain simple scalar configuration values.

    Raises:
        CheckpointFormatException: If any validated field is present with an invalid type.
    """
    if not isinstance(config, dict):
        raise CheckpointFormatException("config must be a dict.")

    required_int_fields = ["sample_rate", "n_fft", "hop_length", "win_length"]
    for key in required_int_fields:
        if key in config and not isinstance(config[key], int):
            raise CheckpointFormatException("config['%s'] must be an int if present." % key)

    if "window" in config and not isinstance(config["window"], str):
        raise CheckpointFormatException("config['window'] must be a str if present.")

    if "center" in config and not isinstance(config["center"], bool):
        raise CheckpointFormatException("config['center'] must be a bool if present.")

    if "stems" in config:
        stems = config["stems"]
        if not isinstance(stems, list) or any(not isinstance(s, str) for s in stems):
            raise CheckpointFormatException("config['stems'] must be list[str] if present.")

    if "waveform_norm" in config and not isinstance(config["waveform_norm"], str):
        raise CheckpointFormatException("config['waveform_norm'] must be a str if present.")

    if "spectrogram_norm" in config and not isinstance(config["spectrogram_norm"], str):
        raise CheckpointFormatException("config['spectrogram_norm'] must be a str if present.")


def validate_checkpoint_extra(extra: dict) -> None:
    """Validate the checkpoint 'extra' payload.

    Backward compatible behavior:
    - Accepts checkpoints where 'extra' is missing or empty.
    - Accepts extra dicts without a 'config' key.
    - If extra['config'] exists, minimally validates it.

    Args:
        extra: The extra payload dict to validate.

    Raises:
        CheckpointFormatException: If extra is not a dict or contains invalid config types.
    """
    if not isinstance(extra, dict):
        raise CheckpointFormatException("extra must be a dict.")

    if "config" in extra:
        cfg = extra["config"]
        if not isinstance(cfg, dict):
            raise CheckpointFormatException("extra['config'] must be a dict if present.")
        _validate_config_dict(cfg)


def build_checkpoint_extra(
    config: Optional[dict] = None,
    model_info: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build a standardized checkpoint extra payload.

    This helper merges optional metadata under stable keys while keeping the
    validator checks centralized.

    Args:
        config: End-to-end config dict (STFT + normalization + stems ordering).
        model_info: Model metadata dict (e.g., stems count, architecture fields).
        extra: Additional user-provided metadata to merge.

    Returns:
        A dict suitable for saving under payload["extra"].

    Raises:
        CheckpointFormatException: If any provided argument has an invalid type or
            fails validation.
    """
    payload: dict = {}

    if extra is not None:
        if not isinstance(extra, dict):
            raise CheckpointFormatException("extra must be a dict if provided.")
        payload.update(extra)

    if config is not None:
        if not isinstance(config, dict):
            raise CheckpointFormatException("config must be a dict if provided.")
        _validate_config_dict(config)
        payload["config"] = dict(config)

    if model_info is not None:
        if not isinstance(model_info, dict):
            raise CheckpointFormatException("model_info must be a dict if provided.")
        payload["model"] = dict(model_info)

    validate_checkpoint_extra(payload)
    return payload


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: Optional[dict] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Output checkpoint path.
        model: Model to serialize.
        optimizer: Optimizer to serialize.
        epoch: Current epoch number.
        step: Global step counter.
        extra: Optional extra payload.

    Raises:
        CheckpointFormatException: If extra is present but invalid.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    extra_payload: dict = extra if extra is not None else {}
    if not isinstance(extra_payload, dict):
        raise CheckpointFormatException("extra must be a dict if provided.")
    validate_checkpoint_extra(extra_payload)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "extra": extra_payload,
    }
    torch.save(payload, str(p))
    logger.info("Saved checkpoint: %s (epoch=%d step=%d)", str(p), int(epoch), int(step))


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> tuple[int, int, dict]:
    """Load a checkpoint into a model (and optionally an optimizer).

    Args:
        path: Path to a checkpoint file.
        model: Model instance to load state into.
        optimizer: Optional optimizer instance to load optimizer_state into.
        map_location: torch.load map_location argument.

    Returns:
        Tuple of (epoch, step, extra).

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        IsADirectoryError: If the path is a directory.
        CheckpointFormatException: If required fields are missing or invalid.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError("Checkpoint not found: %s" % str(p))
    if p.is_dir():
        raise IsADirectoryError("Expected a checkpoint file, got a directory: %s" % str(p))

    payload = torch.load(str(p), map_location=map_location)
    if not isinstance(payload, dict):
        raise CheckpointFormatException("Checkpoint must be a dict payload.")
    if "model_state" not in payload:
        raise CheckpointFormatException("Checkpoint dict missing 'model_state'.")

    model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    epoch = int(payload.get("epoch", 0))
    step = int(payload.get("step", 0))

    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}

    validate_checkpoint_extra(extra)

    logger.info("Loaded checkpoint: %s (epoch=%d step=%d)", str(p), epoch, step)
    return epoch, step, extra


def export_torchscript(
    path: str,
    model: torch.nn.Module,
) -> None:
    """Export a TorchScript model for inference.

    The exported module accepts an input of shape [B, 1, F, T] and returns
    [B, S, F, T]. Scripting is used (not tracing) so T can vary.

    Args:
        path: Output .pt file path.
        model: Model to export.

    Raises:
        CheckpointFormatException: If the model has no parameters (cannot restore device).
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    was_training = bool(model.training)

    try:
        first_param = next(model.parameters())
        original_device = first_param.device
    except StopIteration as e:
        raise CheckpointFormatException(
            "Model has no parameters; cannot export TorchScript."
        ) from e

    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.script(model_cpu)
    scripted.save(str(p))

    model.to(original_device)
    model.train(was_training)

    logger.info("Exported TorchScript model: %s", str(p))
