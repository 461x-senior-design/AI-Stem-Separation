# src/stemmy/inference.py
"""Model inference utilities for stem separation.

This module provides a stable, end-to-end inference path that:
1) Loads a trained model checkpoint (TorchScript .pt or training .pth dict)
2) Constructs an InferenceConfig (optionally derived from checkpoint metadata)
3) Runs preprocessing to produce a model input tensor [B, 1, F, T]
4) Runs the model to predict per-stem ratio masks [B, S, F, T]
5) Optionally renormalizes masks so they sum-to-one across stems
6) Runs postprocessing to reconstruct time-domain stems and optionally export files

Critical project invariant:
- STFT framing settings must match across training and inference.
  In particular, the `center` setting must be consistent with `stemmy.constants.STFT_CENTER`.
"""

import contextlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import torch

from stemmy.constants import (
    DEFAULT_SPECTROGRAM_NORM,
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STEMS_4,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    WINDOW,
)
from stemmy.logging_config import get_logger
from stemmy.models.unet_2d import UNet2D
from stemmy.postprocessing.pipeline import Postprocessor
from stemmy.preprocessing.pipeline import Preprocessor
from stemmy.preprocessing.utility.audio_file_validator import AudioFileValidator

logger = get_logger(__name__)


class InferenceException(Exception):
    """Exception raised for predictable inference failures."""

    pass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for running separation inference.

    This config intentionally mirrors the STFT + normalization knobs used in training.
    If you build this from a checkpoint (config_from_checkpoint), it helps prevent
    training/inference drift in STFT settings and normalization.
    """

    # STFT configuration (must match training/reconstruction expectations)
    sample_rate: int = TARGET_SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    window: str = WINDOW
    center: bool = STFT_CENTER

    # Preprocessing normalization defaults
    waveform_norm: str = DEFAULT_WAVEFORM_NORM
    spectrogram_norm: str = DEFAULT_SPECTROGRAM_NORM

    # Numerical stability (used for renorm + any downstream safe-divisions)
    eps: float = 1e-8

    # Device used for model execution (string to keep config serializable)
    device: str = "cpu"

    # Stem ordering is important: it must match training target ordering
    stems: Optional[list[str]] = None

    # Output behavior controls
    export_files: bool = True
    renorm_masks: bool = True
    validate_outputs: bool = True

    # Optional low-memory inference controls (time-chunked model execution)
    chunk_frames: int = 0
    overlap_frames: int = 0

    # Optional mixed precision inference control (only applicable on CUDA)
    amp: bool = False


@dataclass(frozen=True)
class StemOutputs:
    """Container for separated outputs.

    Stable API:
      waveforms: mapping stem -> waveform (implementation-defined array/tensor)
      paths: mapping stem -> exported file path (empty if export disabled)
      sample_rate: integer sample rate for returned waveforms

    Compatibility:
      Some older scripts expect `stem_waveforms` and `stem_paths`. Provide those
      as read-only properties mapped onto the stable fields.
    """

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
    """Ensure cfg.stems is populated.

    We keep InferenceConfig immutable (frozen dataclass). If cfg.stems is None,
    create a new InferenceConfig with default stems filled in.
    """
    if cfg.stems is None:
        return InferenceConfig(
            sample_rate=cfg.sample_rate,
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

    dev = torch.device(str(device))
    model = UNet2D(stems=int(stems), base_channels=int(base_channels)).to(dev)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        raise InferenceException(
            "State dict mismatch. missing=%s, unexpected=%s" % (str(missing), str(unexpected))
        )

    model.eval()
    logger.info(f"Loaded model checkpoint: {path}")
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


def config_from_checkpoint(ckpt: Mapping[str, Any]) -> InferenceConfig:
    """Build an InferenceConfig from a training checkpoint dict."""
    cfg = _parse_checkpoint_config(ckpt)

    stems_val = cfg.get("stems", None)
    stems_list: Optional[list[str]] = None
    if isinstance(stems_val, list) and all(isinstance(s, str) for s in stems_val):
        stems_list = list(stems_val)

    inferred = InferenceConfig(
        sample_rate=_coerce_int(cfg.get("sample_rate", TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE),
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
        renorm_masks=_coerce_bool(cfg.get("renorm_masks", True), True),
        validate_outputs=_coerce_bool(cfg.get("validate_outputs", True), True),
        chunk_frames=_coerce_int(cfg.get("chunk_frames", 0), 0),
        overlap_frames=_coerce_int(cfg.get("overlap_frames", 0), 0),
        amp=_coerce_bool(cfg.get("amp", False), False),
    )
    inferred = _ensure_cfg_stems(inferred)

    logger.info(
        "Inference config from checkpoint: "
        f"sr={inferred.sample_rate} n_fft={inferred.n_fft} hop={inferred.hop_length} "
        f"win={inferred.win_length} window={inferred.window} center={inferred.center} "
        f"stems={','.join(inferred.stems or [])} device={inferred.device}"
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
    _maybe_check_int("n_fft", cfg.n_fft)
    _maybe_check_int("hop_length", cfg.hop_length)
    _maybe_check_int("win_length", cfg.win_length)
    _maybe_check_str("window", cfg.window)
    _maybe_check_bool("center", cfg.center)
    _maybe_check_str("waveform_norm", cfg.waveform_norm)
    _maybe_check_str("spectrogram_norm", cfg.spectrogram_norm)

    ckpt_stems = ckpt_cfg.get("stems", None)
    if isinstance(ckpt_stems, list) and all(isinstance(s, str) for s in ckpt_stems):
        if list(ckpt_stems) != list(stems):
            raise InferenceException(
                "Checkpoint/runtime stems ordering mismatch: ckpt=%s runtime=%s"
                % (",".join(ckpt_stems), ",".join(list(stems)))
            )


def _validate_input_audio(audio_path: Path) -> None:
    """Run audio validation before preprocessing."""
    validator = AudioFileValidator(str(audio_path))
    is_valid, message = validator.validate()
    if not is_valid:
        raise InferenceException(message)


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
    """Construct a Preprocessor instance using runtime config.

    This filters kwargs to whatever Preprocessor.__init__ actually accepts in this repo.
    """
    kwargs: dict[str, Any] = {
        "sample_rate": cfg.sample_rate,
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


def _preprocess_audio(pre: Preprocessor, audio_path: Path) -> tuple[torch.Tensor, Any]:
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
            f"Preprocessor returned an unsupported output type: {type(out).__name__}"
        )

    if not isinstance(input_tensor, torch.Tensor):
        raise InferenceException("Preprocessor did not return a torch.Tensor input tensor.")
    if input_tensor.ndim != 4:
        raise InferenceException("Preprocessor output must have shape [B, 1, F, T].")
    if input_tensor.shape[1] != 1:
        raise InferenceException(
            "Preprocessor output must have shape [B, 1, F, T] (mono channel dimension). "
            f"Got shape: {tuple(input_tensor.shape)}"
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

    raise InferenceException(f"Unsupported postprocessor result type: {type(result).__name__}")


def _validate_stem_outputs(outputs: StemOutputs, stems: Sequence[str], export_files: bool) -> None:
    """Validate the stable separation output contract."""
    if not isinstance(outputs.sample_rate, int) or outputs.sample_rate <= 0:
        raise InferenceException(f"Invalid sample_rate in outputs: {outputs.sample_rate!r}")

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
            p = outputs.paths.get(stem)
            if not isinstance(p, Path):
                raise InferenceException(
                    "Invalid path type in outputs.paths for stem %s: %s" % (stem, type(p).__name__)
                )
            if not p.exists():
                raise InferenceException(
                    "Export path does not exist for stem %s: %s" % (stem, str(p))
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

            outs: list[torch.Tensor] = []
            for start in range(0, t_frames, step):
                end = min(start + chunk, t_frames)
                x_chunk = input_tensor[..., start:end]
                out_chunk = model(x_chunk)
                if not isinstance(out_chunk, torch.Tensor):
                    raise InferenceException("Model did not return a torch.Tensor.")
                outs.append(out_chunk)

            out_full = torch.cat(outs, dim=-1)

            expected = t_frames
            got = int(out_full.shape[-1])
            if got < expected:
                raise InferenceException(
                    "Chunked inference produced too few frames: expected %d got %d."
                    % (expected, got)
                )
            if got > expected:
                out_full = out_full[..., :expected]
            return out_full


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

    audio_path_p = Path(audio_path).expanduser().resolve()
    output_dir_p = Path(output_dir).expanduser().resolve()

    _validate_input_audio(audio_path_p)

    stem_list = stems if stems is not None else list(cfg.stems or [])
    if len(stem_list) == 0:
        raise ValueError("stems cannot be empty.")

    if export_files is None:
        export_files = bool(cfg.export_files)

    if bool(export_files):
        output_dir_p.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        _validate_checkpoint_alignment(checkpoint, cfg, stem_list)

    pre = _build_preprocessor(cfg)
    post = Postprocessor()

    input_tensor, metadata = _preprocess_audio(pre, audio_path_p)

    device = torch.device(str(cfg.device))
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    model_output = _run_model_forward(model, input_tensor, device, cfg)

    if model_output.ndim != 4:
        raise InferenceException("Model output must have shape [B, S, F, T].")
    if model_output.shape[0] != input_tensor.shape[0]:
        raise InferenceException("Model batch dimension does not match input batch dimension.")
    if (
        model_output.shape[2] != input_tensor.shape[2]
        or model_output.shape[3] != input_tensor.shape[3]
    ):
        raise InferenceException("Model output [F, T] does not match input [F, T].")
    if model_output.shape[1] != len(stem_list):
        raise InferenceException(
            "Stem count mismatch: model returned S=%d, stems list has %d."
            % (int(model_output.shape[1]), int(len(stem_list)))
        )

    # Training optimizes KL(target || softmax(logits)); inference must mirror that.
    model_output = torch.softmax(model_output, dim=1)
    model_output = torch.clamp(model_output, 0.0, 1.0)
    if cfg.renorm_masks:
        model_output = _renorm_sum_to_one(model_output, eps=cfg.eps)

    stem_name = audio_path_p.stem
    logger.info(
        f"Separating audio file: {audio_path_p} -> output_dir={output_dir_p} "
        f"export={bool(export_files)} stems={','.join(stem_list)}"
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
    return outputs


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
