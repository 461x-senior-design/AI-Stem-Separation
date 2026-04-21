# src/stemmy/inference.py
"""Model inference utilities for stem separation.

This module provides a stable, end-to-end inference path that:
1) Loads a trained model checkpoint (TorchScript .pt or training .pth dict)
2) Constructs an InferenceConfig (optionally derived from checkpoint metadata)
3) Runs preprocessing to produce a model input tensor [B, 2, F, T]
4) Runs the model to predict per-stem masks [B, S, 2, F, T]
5) Applies the configured prediction activation
6) Optionally renormalizes masks when that mode is compatible
7) Runs postprocessing to reconstruct time-domain stems and optionally export files

Critical project invariant:
- STFT framing settings must match across training and inference.
  In particular, the `center` setting must be consistent with
  `stemmy.constants.STFT_CENTER`.
"""

import contextlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import torch

from stemmy.constants import (
    BOLD_PURPLE,
    DEFAULT_LOSS_MODE,
    DEFAULT_PRED_ACTIVATION,
    DEFAULT_RENORM_MASKS,
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_TARGET_MODE,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STEMS_4,
    STFT_CENTER,
    SUPPORTED_LOSS_MODES,
    SUPPORTED_PRED_ACTIVATIONS,
    SUPPORTED_TARGET_MODES,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    WINDOW,
)
from stemmy.logging_config import get_logger
from stemmy.models.unet_2d import UNet2D
from stemmy.postprocessing.pipeline import Postprocessor
from stemmy.preprocessing.pipeline import Preprocessor
from stemmy.tool.progress_theme import create_themed_progress, start_eq_animator

logger = get_logger(__name__)


class InferenceException(Exception):
    """Exception raised for predictable inference failures."""

    pass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for running separation inference."""

    sample_rate: int = TARGET_SAMPLE_RATE
    audio_channels: int = TARGET_CHANNELS
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    window: str = WINDOW
    center: bool = STFT_CENTER

    waveform_norm: str = DEFAULT_WAVEFORM_NORM
    spectrogram_norm: str = DEFAULT_SPECTROGRAM_NORM

    eps: float = 1e-8
    device: str = "cpu"
    stems: Optional[list[str]] = None

    export_files: bool = True
    renorm_masks: bool = DEFAULT_RENORM_MASKS
    validate_outputs: bool = True

    chunk_frames: int = 0
    overlap_frames: int = 0
    amp: bool = False

    target_mode: str = DEFAULT_TARGET_MODE
    pred_activation: str = DEFAULT_PRED_ACTIVATION
    loss_mode: str = DEFAULT_LOSS_MODE


@dataclass(frozen=True)
class StemOutputs:
    """Container for separated outputs."""

    waveforms: Mapping[str, Any]
    paths: Mapping[str, Path]
    sample_rate: int

    @property
    def stem_waveforms(self) -> Mapping[str, Any]:
        """Backward-compatible alias for scripts expecting `stem_waveforms`."""
        return self.waveforms

    @property
    def stem_paths(self) -> Mapping[str, Path]:
        """Backward-compatible alias for scripts expecting `stem_paths`."""
        return self.paths


def _default_stems() -> list[str]:
    """Return the project default stem ordering (copy)."""
    return list(STEMS_4)


def _ensure_cfg_stems(cfg: InferenceConfig) -> InferenceConfig:
    """Ensure cfg.stems is populated."""
    if cfg.stems is None:
        return InferenceConfig(
            sample_rate=cfg.sample_rate,
            audio_channels=cfg.audio_channels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=cfg.window,
            center=cfg.center,
            waveform_norm=cfg.waveform_norm,
            spectrogram_norm=cfg.spectrogram_norm,
            eps=cfg.eps,
            device=cfg.device,
            stems=_default_stems(),
            export_files=cfg.export_files,
            renorm_masks=cfg.renorm_masks,
            validate_outputs=cfg.validate_outputs,
            chunk_frames=cfg.chunk_frames,
            overlap_frames=cfg.overlap_frames,
            amp=cfg.amp,
            target_mode=cfg.target_mode,
            pred_activation=cfg.pred_activation,
            loss_mode=cfg.loss_mode,
        )
    return cfg


def load_torchscript_model(
    checkpoint_path: Union[str, Path], device: str = "cpu"
) -> torch.jit.ScriptModule:
    """Load a TorchScript model (.pt) and set eval mode."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = torch.device(str(device))
    model = torch.jit.load(str(path), map_location=dev)
    model.eval()
    return model


def load_pth_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    stems: int = 4,
    base_channels: Optional[int] = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a training checkpoint dict (.pth) containing 'model_state'."""
    if not isinstance(stems, int) or stems <= 0:
        raise ValueError("stems must be a positive int")

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a .pth file, got a directory: {path}")

    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise InferenceException("Checkpoint must be a dict.")
    if "model_state" not in ckpt:
        raise InferenceException("Checkpoint dict missing 'model_state'.")

    ckpt_cfg = _parse_checkpoint_config(ckpt)

    if base_channels is None:
        cfg_base_channels = ckpt_cfg.get("base_channels", None)
        if isinstance(cfg_base_channels, int) and cfg_base_channels > 0:
            base_channels = int(cfg_base_channels)
        else:
            base_channels = 64

    if not isinstance(base_channels, int) or base_channels <= 0:
        raise ValueError("base_channels must be a positive int")

    cfg_audio_channels = ckpt_cfg.get("audio_channels", TARGET_CHANNELS)
    try:
        audio_channels = int(cfg_audio_channels)
    except (TypeError, ValueError) as exc:
        raise InferenceException(
            "Invalid checkpoint audio_channels value: %r" % (cfg_audio_channels,)
        ) from exc
    if audio_channels <= 0:
        raise InferenceException("audio_channels must be positive.")

    dev = torch.device(str(device))
    model = UNet2D(
        stems=int(stems),
        base_channels=int(base_channels),
        audio_channels=int(audio_channels),
    ).to(dev)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        raise InferenceException(
            "State dict mismatch. missing=%s, unexpected=%s" % (str(missing), str(unexpected))
        )

    model.eval()
    logger.info("Loaded model checkpoint: %s", str(path))
    return model, ckpt


def _parse_checkpoint_config(ckpt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the training config dict from a checkpoint if present."""
    if not isinstance(ckpt, Mapping):
        return {}
    extra = ckpt.get("extra", {})
    if not isinstance(extra, Mapping):
        return {}
    cfg = extra.get("config", {})
    if not isinstance(cfg, Mapping):
        return {}
    return cfg


def _coerce_bool(value: Any, default: bool) -> bool:
    """Best-effort coercion to bool, with a default fallback."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort coercion to int, with a default fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str) -> str:
    """Best-effort coercion to str, with a default fallback."""
    if value is None:
        return default
    return str(value)


def _validate_cfg_modes(cfg: InferenceConfig) -> None:
    """Validate target/activation/loss modes."""
    if cfg.target_mode not in SUPPORTED_TARGET_MODES:
        raise InferenceException("Unsupported target_mode: %s" % cfg.target_mode)
    if cfg.pred_activation not in SUPPORTED_PRED_ACTIVATIONS:
        raise InferenceException("Unsupported pred_activation: %s" % cfg.pred_activation)
    if cfg.loss_mode not in SUPPORTED_LOSS_MODES:
        raise InferenceException("Unsupported loss_mode: %s" % cfg.loss_mode)

    if not isinstance(cfg.audio_channels, int) or cfg.audio_channels <= 0:
        raise InferenceException("audio_channels must be a positive int.")


def config_from_checkpoint(ckpt: Mapping[str, Any]) -> InferenceConfig:
    """Build an InferenceConfig from a training checkpoint dict."""
    cfg = _parse_checkpoint_config(ckpt)

    stems_val = cfg.get("stems", None)
    stems_list: Optional[list[str]] = None
    if isinstance(stems_val, list) and all(isinstance(s, str) for s in stems_val):
        stems_list = list(stems_val)

    inferred = InferenceConfig(
        sample_rate=_coerce_int(cfg.get("sample_rate", TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE),
        audio_channels=_coerce_int(cfg.get("audio_channels", TARGET_CHANNELS), TARGET_CHANNELS),
        n_fft=_coerce_int(cfg.get("n_fft", N_FFT), N_FFT),
        hop_length=_coerce_int(cfg.get("hop_length", HOP_LENGTH), HOP_LENGTH),
        win_length=_coerce_int(cfg.get("win_length", WIN_LENGTH), WIN_LENGTH),
        window=_coerce_str(cfg.get("window", WINDOW), WINDOW),
        center=_coerce_bool(cfg.get("center", STFT_CENTER), STFT_CENTER),
        waveform_norm=_coerce_str(
            cfg.get("waveform_norm", DEFAULT_WAVEFORM_NORM), DEFAULT_WAVEFORM_NORM
        ),
        spectrogram_norm=_coerce_str(
            cfg.get("spectrogram_norm", DEFAULT_SPECTROGRAM_NORM),
            DEFAULT_SPECTROGRAM_NORM,
        ),
        eps=float(cfg.get("eps", 1e-8)) if cfg.get("eps", None) is not None else 1e-8,
        device=_coerce_str(cfg.get("device", "cpu"), "cpu"),
        stems=stems_list,
        export_files=_coerce_bool(cfg.get("export_files", True), True),
        renorm_masks=_coerce_bool(
            cfg.get("renorm_masks", DEFAULT_RENORM_MASKS), DEFAULT_RENORM_MASKS
        ),
        validate_outputs=_coerce_bool(cfg.get("validate_outputs", True), True),
        chunk_frames=_coerce_int(cfg.get("chunk_frames", 0), 0),
        overlap_frames=_coerce_int(cfg.get("overlap_frames", 0), 0),
        amp=_coerce_bool(cfg.get("amp", False), False),
        target_mode=_coerce_str(cfg.get("target_mode", DEFAULT_TARGET_MODE), DEFAULT_TARGET_MODE),
        pred_activation=_coerce_str(
            cfg.get("pred_activation", DEFAULT_PRED_ACTIVATION),
            DEFAULT_PRED_ACTIVATION,
        ),
        loss_mode=_coerce_str(cfg.get("loss_mode", DEFAULT_LOSS_MODE), DEFAULT_LOSS_MODE),
    )
    inferred = _ensure_cfg_stems(inferred)
    _validate_cfg_modes(inferred)

    logger.info(
        "Inference config from checkpoint: "
        "sr=%s channels=%s n_fft=%s hop=%s win=%s window=%s center=%s "
        "stems=%s device=%s target_mode=%s pred_activation=%s loss_mode=%s",
        inferred.sample_rate,
        inferred.audio_channels,
        inferred.n_fft,
        inferred.hop_length,
        inferred.win_length,
        inferred.window,
        inferred.center,
        ",".join(inferred.stems or []),
        inferred.device,
        inferred.target_mode,
        inferred.pred_activation,
        inferred.loss_mode,
    )
    return inferred


def _validate_checkpoint_alignment(
    ckpt: Mapping[str, Any],
    cfg: InferenceConfig,
    stems: Sequence[str],
) -> None:
    """Fail fast if checkpoint config conflicts with runtime config."""
    ckpt_cfg = _parse_checkpoint_config(ckpt)
    if not ckpt_cfg:
        return

    def _maybe_check_int(key: str, runtime_val: int) -> None:
        if key not in ckpt_cfg:
            return
        ckpt_val = ckpt_cfg.get(key)
        if isinstance(ckpt_val, int) and int(ckpt_val) != int(runtime_val):
            raise InferenceException(
                "Checkpoint/runtime mismatch for %s: ckpt=%s runtime=%s"
                % (key, str(ckpt_val), str(runtime_val))
            )

    def _maybe_check_bool(key: str, runtime_val: bool) -> None:
        if key not in ckpt_cfg:
            return
        ckpt_val = ckpt_cfg.get(key)
        if isinstance(ckpt_val, bool) and bool(ckpt_val) != bool(runtime_val):
            raise InferenceException(
                "Checkpoint/runtime mismatch for %s: ckpt=%s runtime=%s"
                % (key, str(ckpt_val), str(runtime_val))
            )

    def _maybe_check_str(key: str, runtime_val: str) -> None:
        if key not in ckpt_cfg:
            return
        ckpt_val = ckpt_cfg.get(key)
        if isinstance(ckpt_val, str) and str(ckpt_val) != str(runtime_val):
            raise InferenceException(
                "Checkpoint/runtime mismatch for %s: ckpt=%s runtime=%s"
                % (key, str(ckpt_val), str(runtime_val))
            )

    _maybe_check_int("sample_rate", cfg.sample_rate)
    _maybe_check_int("audio_channels", cfg.audio_channels)
    _maybe_check_int("n_fft", cfg.n_fft)
    _maybe_check_int("hop_length", cfg.hop_length)
    _maybe_check_int("win_length", cfg.win_length)
    _maybe_check_str("window", cfg.window)
    _maybe_check_bool("center", cfg.center)
    _maybe_check_str("waveform_norm", cfg.waveform_norm)
    _maybe_check_str("spectrogram_norm", cfg.spectrogram_norm)
    _maybe_check_str("target_mode", cfg.target_mode)
    _maybe_check_str("pred_activation", cfg.pred_activation)
    _maybe_check_str("loss_mode", cfg.loss_mode)

    ckpt_stems = ckpt_cfg.get("stems", None)
    if isinstance(ckpt_stems, list) and all(isinstance(s, str) for s in ckpt_stems):
        if list(ckpt_stems) != list(stems):
            raise InferenceException(
                "Checkpoint/runtime stems ordering mismatch: ckpt=%s runtime=%s"
                % (",".join(ckpt_stems), ",".join(list(stems)))
            )


def _renorm_sum_to_one(masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize masks so they sum to 1 across the stem dimension."""
    denom = masks.sum(dim=1, keepdim=True).clamp_min(float(eps))
    return masks / denom


def _call_with_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> Any:
    """Call a function/constructor with only the kwargs it supports."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(**kwargs)

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return callable_obj(**kwargs)

    allowed = {name for name in params.keys() if name != "self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return callable_obj(**filtered)


def _build_preprocessor(cfg: InferenceConfig) -> Preprocessor:
    """Construct a Preprocessor instance using runtime config."""
    kwargs: dict[str, Any] = {
        "sample_rate": cfg.sample_rate,
        "audio_channels": cfg.audio_channels,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "win_length": cfg.win_length,
        "window": cfg.window,
        "center": cfg.center,
        "waveform_norm": cfg.waveform_norm,
        "spectrogram_norm": cfg.spectrogram_norm,
        "eps": cfg.eps,
    }
    try:
        return _call_with_supported_kwargs(Preprocessor, kwargs)
    except TypeError as exc:
        raise InferenceException("Failed to construct Preprocessor: %s" % str(exc)) from exc


def _preprocess_audio(
    pre: Preprocessor, audio_path: Path, cfg: InferenceConfig
) -> tuple[torch.Tensor, Any]:
    """Run the preprocessing pipeline and normalize expected return types."""
    try:
        out = pre.process(audio_path)
    except TypeError:
        out = pre.process(str(audio_path))

    if isinstance(out, tuple) and len(out) == 2:
        input_tensor, metadata = out
    elif isinstance(out, dict) and "input_tensor" in out:
        input_tensor = out["input_tensor"]
        metadata = out.get("metadata", None)
    else:
        raise InferenceException(
            "Preprocessor returned an unsupported output type: %s" % type(out).__name__
        )

    if not isinstance(input_tensor, torch.Tensor):
        raise InferenceException("Preprocessor did not return a torch.Tensor input tensor.")
    if input_tensor.ndim != 4:
        raise InferenceException("Preprocessor output must have shape [B, C, F, T].")
    if int(input_tensor.shape[1]) != int(cfg.audio_channels):
        raise InferenceException(
            "Preprocessor output must have shape [B, %d, F, T]. Got shape: %s"
            % (int(cfg.audio_channels), str(tuple(input_tensor.shape)))
        )

    return input_tensor, metadata


def _postprocess_masks(
    post: Postprocessor,
    model_output: torch.Tensor,
    metadata: Any,
    input_tensor: torch.Tensor,
    output_dir: Path,
    stem_name: str,
    export_files: bool,
    stems: Sequence[str],
) -> Any:
    """Run the postprocessing pipeline using a compatible call signature."""
    try:
        return post.process(
            model_output=model_output,
            metadata=metadata,
            input_tensor=input_tensor,
            output_dir=output_dir,
            stem_name=stem_name,
            export_files=bool(export_files),
            stems=list(stems),
        )
    except TypeError:
        return post.process(
            model_output=model_output,
            metadata=metadata,
            output_dir=output_dir,
            export_files=bool(export_files),
            stems=list(stems),
        )


def _adapt_postprocess_result(result: Any) -> StemOutputs:
    """Convert a pipeline-specific result into the stable StemOutputs container."""
    if hasattr(result, "waveforms") and hasattr(result, "paths") and hasattr(result, "sample_rate"):
        waveforms = getattr(result, "waveforms")
        paths = getattr(result, "paths")
        sample_rate = getattr(result, "sample_rate")

        paths_map = dict(paths) if isinstance(paths, Mapping) else {}
        return StemOutputs(waveforms=waveforms, paths=paths_map, sample_rate=int(sample_rate))

    if (
        hasattr(result, "stem_waveforms")
        and hasattr(result, "stem_paths")
        and hasattr(result, "sample_rate")
    ):
        waveforms = getattr(result, "stem_waveforms")
        paths = getattr(result, "stem_paths")
        sample_rate = getattr(result, "sample_rate")

        paths_map = dict(paths) if isinstance(paths, Mapping) else {}
        return StemOutputs(waveforms=waveforms, paths=paths_map, sample_rate=int(sample_rate))

    if isinstance(result, dict):
        if "waveforms" in result and "paths" in result and "sample_rate" in result:
            waveforms = result.get("waveforms", {})
            paths = result.get("paths", {})
            sample_rate = result.get("sample_rate", None)
            if sample_rate is None:
                raise InferenceException("Postprocessor result dict missing 'sample_rate'.")
            return StemOutputs(
                waveforms=waveforms,
                paths=dict(paths) if isinstance(paths, Mapping) else {},
                sample_rate=int(sample_rate),
            )

        if "stem_waveforms" in result and "stem_paths" in result and "sample_rate" in result:
            waveforms = result.get("stem_waveforms", {})
            paths = result.get("stem_paths", {})
            sample_rate = result.get("sample_rate", None)
            if sample_rate is None:
                raise InferenceException("Postprocessor result dict missing 'sample_rate'.")
            return StemOutputs(
                waveforms=waveforms,
                paths=dict(paths) if isinstance(paths, Mapping) else {},
                sample_rate=int(sample_rate),
            )

    raise InferenceException("Unsupported postprocessor result type: %s" % type(result).__name__)


def _validate_stem_outputs(outputs: StemOutputs, stems: Sequence[str], export_files: bool) -> None:
    """Validate the stable separation output contract."""
    if not isinstance(outputs.sample_rate, int) or outputs.sample_rate <= 0:
        raise InferenceException("Invalid sample_rate in outputs: %r" % (outputs.sample_rate,))

    if not isinstance(outputs.waveforms, Mapping):
        raise InferenceException("outputs.waveforms must be a mapping stem -> waveform")

    stems_set = set(stems)
    waveforms_keys = set(outputs.waveforms.keys())
    missing_waveforms = [s for s in stems if s not in outputs.waveforms]
    if missing_waveforms:
        raise InferenceException(
            "Missing stems in outputs.waveforms: %s" % (",".join(missing_waveforms))
        )
    extra_waveforms = sorted(waveforms_keys - stems_set)
    if extra_waveforms:
        raise InferenceException(
            "Unexpected stems in outputs.waveforms: %s" % (",".join(extra_waveforms))
        )

    if export_files:
        if not isinstance(outputs.paths, Mapping):
            raise InferenceException("outputs.paths must be a mapping stem -> Path when exporting")

        paths_keys = set(outputs.paths.keys())
        missing_paths = [s for s in stems if s not in outputs.paths]
        if missing_paths:
            raise InferenceException(
                "Missing stems in outputs.paths: %s" % (",".join(missing_paths))
            )
        extra_paths = sorted(paths_keys - stems_set)
        if extra_paths:
            raise InferenceException(
                "Unexpected stems in outputs.paths: %s" % (",".join(extra_paths))
            )

        for stem in stems:
            path = outputs.paths.get(stem)
            if not isinstance(path, Path):
                raise InferenceException(
                    "Invalid path type in outputs.paths for stem %s: %s"
                    % (stem, type(path).__name__)
                )
            if not path.exists():
                raise InferenceException(
                    "Export path does not exist for stem %s: %s" % (stem, str(path))
                )
    else:
        if isinstance(outputs.paths, Mapping) and len(outputs.paths) != 0:
            raise InferenceException("export_files=False but outputs.paths is not empty.")


def _infer_autocast_context(device: torch.device, amp: bool) -> Any:
    """Return an autocast context appropriate for the device."""
    if not amp:
        return contextlib.nullcontext()

    if device.type != "cuda":
        return contextlib.nullcontext()

    try:
        return torch.amp.autocast(device_type="cuda")
    except AttributeError:
        return torch.cuda.amp.autocast()


def _validate_chunk_settings(cfg: InferenceConfig, t_frames: int) -> tuple[int, int]:
    """Validate and normalize chunking settings for model execution."""
    chunk = int(cfg.chunk_frames)
    overlap = int(cfg.overlap_frames)

    if chunk <= 0:
        return 0, 0

    if t_frames <= 0:
        raise InferenceException("Invalid input tensor time dimension (T <= 0).")

    if chunk < 1:
        raise InferenceException("chunk_frames must be >= 1 when enabled.")
    if overlap < 0:
        raise InferenceException("overlap_frames must be >= 0.")
    if overlap >= chunk:
        raise InferenceException("overlap_frames must be < chunk_frames.")
    if chunk > t_frames:
        chunk = t_frames
    return chunk, overlap


def _run_model_forward(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    cfg: InferenceConfig,
) -> torch.Tensor:
    """Run model forward, optionally with chunking and AMP."""
    t_frames = int(input_tensor.shape[-1])
    chunk, overlap = _validate_chunk_settings(cfg, t_frames)

    with torch.inference_mode():
        with _infer_autocast_context(device, bool(cfg.amp)):
            if chunk <= 0:
                out = model(input_tensor)
                if not isinstance(out, torch.Tensor):
                    raise InferenceException("Model did not return a torch.Tensor.")
                return out

            step = chunk - overlap
            if step <= 0:
                raise InferenceException("Invalid chunk/overlap configuration.")

            out_full: Optional[torch.Tensor] = None
            weight_full: Optional[torch.Tensor] = None

            for start in range(0, t_frames, step):
                end = min(start + chunk, t_frames)
                x_chunk = input_tensor[..., start:end]

                out_chunk = model(x_chunk)
                if not isinstance(out_chunk, torch.Tensor):
                    raise InferenceException("Model did not return a torch.Tensor.")
                if out_chunk.ndim != 5:
                    raise InferenceException(
                        "Chunked model output must have shape [B, S, C, F, T]."
                    )

                if out_full is None:
                    out_full = torch.zeros(
                        (
                            out_chunk.shape[0],
                            out_chunk.shape[1],
                            out_chunk.shape[2],
                            out_chunk.shape[3],
                            t_frames,
                        ),
                        device=out_chunk.device,
                        dtype=out_chunk.dtype,
                    )
                    weight_full = torch.zeros(
                        (t_frames,), device=out_chunk.device, dtype=torch.float32
                    )

                length = int(end - start)

                if overlap <= 0:
                    weights = torch.ones((length,), device=out_chunk.device, dtype=torch.float32)
                else:
                    weights = torch.ones((length,), device=out_chunk.device, dtype=torch.float32)

                    if start > 0:
                        n = min(overlap, length)
                        if n > 0:
                            ramp = torch.arange(
                                1,
                                n + 1,
                                device=out_chunk.device,
                                dtype=torch.float32,
                            ) / float(n + 1)
                            weights[:n] = ramp

                    if end < t_frames:
                        n = min(overlap, length)
                        if n > 0:
                            ramp = torch.arange(
                                n,
                                0,
                                -1,
                                device=out_chunk.device,
                                dtype=torch.float32,
                            ) / float(n + 1)
                            weights[length - n : length] = torch.minimum(
                                weights[length - n : length],
                                ramp,
                            )

                out_full[..., start:end] += out_chunk * weights.to(dtype=out_chunk.dtype).view(
                    1, 1, 1, 1, -1
                )
                weight_full[start:end] += weights

            if out_full is None or weight_full is None:
                raise InferenceException("Chunked inference produced no output.")

            denom = (
                weight_full.clamp_min(float(cfg.eps))
                .to(device=out_full.device, dtype=out_full.dtype)
                .view(1, 1, 1, 1, -1)
            )
            out_full = out_full / denom

            got = int(out_full.shape[-1])
            if got != t_frames:
                raise InferenceException(
                    "Chunked inference produced unexpected frame count: expected %d got %d."
                    % (t_frames, got)
                )

            return out_full


def _apply_prediction_activation(model_output: torch.Tensor, cfg: InferenceConfig) -> torch.Tensor:
    """Apply the configured prediction activation to raw model outputs."""
    if cfg.pred_activation == "softmax":
        masks = torch.softmax(model_output, dim=1)
        masks = torch.clamp(masks, 0.0, 1.0)
        if cfg.renorm_masks:
            masks = _renorm_sum_to_one(masks, eps=cfg.eps)
        return masks

    if cfg.pred_activation == "sigmoid":
        masks = torch.sigmoid(model_output)
        masks = torch.clamp(masks, 0.0, 1.0)
        return masks

    raise InferenceException("Unsupported pred_activation: %s" % cfg.pred_activation)


def separate_audio_file(
    audio_path: Union[str, Path],
    model: torch.nn.Module,
    cfg: InferenceConfig,
    output_dir: Union[str, Path],
    export_files: Optional[bool] = None,
    stems: Optional[list[str]] = None,
    checkpoint: Optional[Mapping[str, Any]] = None,
) -> StemOutputs:
    """Run end-to-end separation on a single audio file."""
    cfg = _ensure_cfg_stems(cfg)
    _validate_cfg_modes(cfg)

    audio_path_p = Path(audio_path).expanduser().resolve()
    output_dir_p = Path(output_dir).expanduser().resolve()

    stem_list = stems if stems is not None else list(cfg.stems or [])
    if len(stem_list) == 0:
        raise ValueError("stems cannot be empty.")

    if export_files is None:
        export_files = bool(cfg.export_files)

    if bool(export_files):
        output_dir_p.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        _validate_checkpoint_alignment(checkpoint, cfg, stem_list)

    if cfg.pred_activation == "sigmoid" and cfg.renorm_masks:
        logger.warning(
            "renorm_masks=True with sigmoid prediction is incompatible with independent masks. "
            "Disabling renorm for this inference call."
        )
        cfg = InferenceConfig(
            sample_rate=cfg.sample_rate,
            audio_channels=cfg.audio_channels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=cfg.window,
            center=cfg.center,
            waveform_norm=cfg.waveform_norm,
            spectrogram_norm=cfg.spectrogram_norm,
            eps=cfg.eps,
            device=cfg.device,
            stems=list(cfg.stems or []),
            export_files=cfg.export_files,
            renorm_masks=False,
            validate_outputs=cfg.validate_outputs,
            chunk_frames=cfg.chunk_frames,
            overlap_frames=cfg.overlap_frames,
            amp=cfg.amp,
            target_mode=cfg.target_mode,
            pred_activation=cfg.pred_activation,
            loss_mode=cfg.loss_mode,
        )

    with create_themed_progress(
        "Separate {task.fields[phase]}", title_style=BOLD_PURPLE
    ) as progress:
        separate_task = progress.add_task(
            "separate",
            total=6,
            phase="",
            eq="",
        )
        anim_stop, anim_thread = start_eq_animator(progress, separate_task, fps=8)
        try:
            progress.update(separate_task, advance=1, phase="Init")

            pre = _build_preprocessor(cfg)
            post = Postprocessor()

            progress.update(separate_task, advance=1, phase="Pre")
            input_tensor, metadata = _preprocess_audio(pre, audio_path_p, cfg)

            progress.update(separate_task, advance=1, phase="Device")
            device = torch.device(str(cfg.device))
            model = model.to(device)
            input_tensor = input_tensor.to(device)

            progress.update(separate_task, advance=1, phase="Infer")
            model_output = _run_model_forward(model, input_tensor, device, cfg)

            if model_output.ndim != 5:
                raise InferenceException("Model output must have shape [B, S, C, F, T].")
            if model_output.shape[0] != input_tensor.shape[0]:
                raise InferenceException(
                    "Model batch dimension does not match input batch dimension."
                )
            if int(model_output.shape[2]) != int(cfg.audio_channels):
                raise InferenceException(
                    "Model output channel dimension does not match cfg.audio_channels."
                )
            if (
                model_output.shape[3] != input_tensor.shape[2]
                or model_output.shape[4] != input_tensor.shape[3]
            ):
                raise InferenceException("Model output [F, T] does not match input [F, T].")
            if model_output.shape[1] != len(stem_list):
                raise InferenceException(
                    "Stem count mismatch: model returned S=%d, stems list has %d."
                    % (int(model_output.shape[1]), int(len(stem_list)))
                )

            model_output = _apply_prediction_activation(model_output, cfg)

            progress.update(separate_task, advance=1, phase="Export")
            stem_name = audio_path_p.stem
            logger.info(
                "Separating audio file: %s -> output_dir=%s export=%s stems=%s",
                str(audio_path_p),
                str(output_dir_p),
                bool(export_files),
                ",".join(stem_list),
            )

            result = _postprocess_masks(
                post=post,
                model_output=model_output.cpu(),
                metadata=metadata,
                input_tensor=input_tensor.cpu(),
                output_dir=output_dir_p,
                stem_name=stem_name,
                export_files=bool(export_files),
                stems=stem_list,
            )
            outputs = _adapt_postprocess_result(result)
            if cfg.validate_outputs:
                _validate_stem_outputs(outputs, stems=stem_list, export_files=bool(export_files))

            progress.update(separate_task, advance=1, phase="Done")
            return outputs
        finally:
            anim_stop.set()
            anim_thread.join()
            progress.update(separate_task, eq="▁▁▁▁▁")


def separate_batch(
    audio_files: Iterable[Union[str, Path]],
    model: torch.nn.Module,
    cfg: InferenceConfig,
    output_dir: Union[str, Path],
    export_files: Optional[bool] = None,
    stems: Optional[list[str]] = None,
    checkpoint: Optional[Mapping[str, Any]] = None,
) -> dict[str, StemOutputs]:
    """Run inference for multiple audio files."""
    results: dict[str, StemOutputs] = {}
    for audio_path in audio_files:
        outputs = separate_audio_file(
            audio_path=audio_path,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            export_files=export_files,
            stems=stems,
            checkpoint=checkpoint,
        )
        results[str(audio_path)] = outputs
    return results
