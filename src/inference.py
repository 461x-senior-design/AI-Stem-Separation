"""Model inference utilities for stem separation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import torch

from src.constants import (
    DEFAULT_WAVEFORM_NORM,
    HOP_LENGTH,
    N_FFT,
    STEMS_4,
    STFT_CENTER,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
)
from src.models.unet_2d import UNet2D
from src.postprocessing.audio import denormalize_waveform, export_audio
from src.postprocessing.spectral import apply_mask, combine_magnitude_phase, compute_istft
from src.postprocessing.utility.output_validator import OutputValidationException, OutputValidator
from src.preprocessing.audio import ensure_stereo, load_audio, normalize_waveform
from src.preprocessing.spectral import compute_stft, split_magnitude_phase
from src.preprocessing.utility.audio_file_validator import (
    AudioFileValidator,
    AudioValidationException,
)
from src.training.stft import freq_minmax_normalize


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference-time preprocessing and reconstruction."""

    sample_rate: int = TARGET_SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    center: bool = STFT_CENTER
    waveform_norm: str = DEFAULT_WAVEFORM_NORM
    eps: float = 1e-8
    device: str = "cpu"
    renorm_masks: bool = True
    validate_outputs: bool = True


@dataclass(frozen=True)
class InferenceArtifacts:
    """Artifacts needed to reconstruct stems from model masks."""

    mix_magnitude: np.ndarray
    mix_phase: np.ndarray
    waveform_norm_params: Mapping[str, float]
    sample_rate: int
    num_samples: int


@dataclass(frozen=True)
class StemOutputs:
    """Container for separated waveforms and optional output paths."""

    waveforms: Mapping[str, np.ndarray]
    paths: Mapping[str, Path]
    sample_rate: int


def load_torchscript_model(
    checkpoint_path: Union[str, Path], device: str = "cpu"
) -> torch.jit.ScriptModule:
    """Load a TorchScript model (.pt)."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = torch.device(device)
    model = torch.jit.load(str(path), map_location=dev)
    model.eval()
    return model


def load_pth_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    stems: int = 4,
) -> tuple[torch.nn.Module, dict]:
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
        raise RuntimeError("Checkpoint must be a dict.")
    if "model_state" not in ckpt:
        raise RuntimeError("Checkpoint dict missing 'model_state'.")

    dev = torch.device(device)
    model = UNet2D(stems=stems).to(dev)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    model.eval()
    return model, ckpt


def config_from_checkpoint(ckpt: Mapping[str, object]) -> InferenceConfig:
    """Build an inference config from a training checkpoint dict."""
    extra = ckpt.get("extra", {}) if isinstance(ckpt, Mapping) else {}
    config = extra.get("config", {}) if isinstance(extra, Mapping) else {}

    def _get_int(key: str, default: int) -> int:
        value = config.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _get_str(key: str, default: str) -> str:
        value = config.get(key, default)
        return str(value) if value is not None else default

    def _get_bool(key: str, default: bool) -> bool:
        value = config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value) if value is not None else default

    return InferenceConfig(
        sample_rate=_get_int("sample_rate", TARGET_SAMPLE_RATE),
        n_fft=_get_int("n_fft", N_FFT),
        hop_length=_get_int("hop_length", HOP_LENGTH),
        win_length=_get_int("win_length", WIN_LENGTH),
        center=_get_bool("center", STFT_CENTER),
        waveform_norm=_get_str("waveform_norm", DEFAULT_WAVEFORM_NORM),
    )


def _validate_input_audio(audio_path: Path) -> None:
    """Run audio validation before loading audio."""
    validator = AudioFileValidator(str(audio_path))
    is_valid, message = validator.validate()
    if not is_valid:
        raise AudioValidationException(message)


def prepare_model_input(
    audio_path: Union[str, Path], cfg: InferenceConfig
) -> tuple[torch.Tensor, InferenceArtifacts]:
    """Load audio, validate, and prepare model input plus reconstruction artifacts."""
    audio_path = Path(audio_path).expanduser().resolve()
    _validate_input_audio(audio_path)

    waveform, _sr = load_audio(audio_path, sr=cfg.sample_rate, mono=False)
    waveform = ensure_stereo(waveform)
    num_samples = int(waveform.shape[-1])

    waveform, waveform_norm_params = normalize_waveform(waveform, method=cfg.waveform_norm)

    stft_channels = [
        compute_stft(
            waveform[ch],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            center=cfg.center,
        )
        for ch in range(waveform.shape[0])
    ]
    stft_mix = np.stack(stft_channels, axis=0)
    mix_magnitude, mix_phase = split_magnitude_phase(stft_mix)

    mono_waveform = waveform.mean(axis=0)
    stft_mono = compute_stft(
        mono_waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        center=cfg.center,
    )
    mono_mag = np.abs(stft_mono)

    mono_mag_t = torch.from_numpy(mono_mag).to(torch.float32)
    mono_mag_norm, _f_min, _f_max = freq_minmax_normalize(mono_mag_t, eps=cfg.eps)

    model_input = mono_mag_norm.unsqueeze(0).unsqueeze(0)

    artifacts = InferenceArtifacts(
        mix_magnitude=mix_magnitude,
        mix_phase=mix_phase,
        waveform_norm_params=waveform_norm_params,
        sample_rate=cfg.sample_rate,
        num_samples=num_samples,
    )
    return model_input, artifacts


def _validate_stem_output(waveform: np.ndarray) -> None:
    """Validate separated waveform."""
    validator = OutputValidator(waveform)
    is_valid, message = validator.validate()
    if not is_valid:
        raise OutputValidationException(message)


def separate_from_model_output(
    model_output: torch.Tensor,
    artifacts: InferenceArtifacts,
    cfg: InferenceConfig,
    stem_names: Sequence[str] = STEMS_4,
    output_dir: Union[str, Path, None] = None,
    track_name: Union[str, None] = None,
    export_files: bool = True,
) -> StemOutputs:
    """Convert model masks into audio waveforms and optionally export WAV files."""
    if model_output.ndim != 4:
        raise ValueError("model_output must have shape [1, S, F, T]")

    masks = torch.clamp(model_output, 0.0, 1.0).squeeze(0).cpu().numpy()
    if masks.shape[1:] != artifacts.mix_magnitude.shape[1:]:
        raise ValueError("model_output frequency/time dimensions must match mix magnitude shape")
    if masks.shape[0] != len(stem_names):
        raise ValueError("stem_names length must match model output stems")

    if cfg.renorm_masks:
        denom = np.maximum(masks.sum(axis=0, keepdims=True), cfg.eps)
        masks = masks / denom

    waveforms: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}

    if output_dir is not None:
        output_dir = Path(output_dir)

    for stem_idx, stem_name in enumerate(stem_names):
        mask = masks[stem_idx]
        stereo_mask = np.stack([mask, mask])

        masked_mag = apply_mask(artifacts.mix_magnitude, stereo_mask)
        stft_complex = combine_magnitude_phase(masked_mag, artifacts.mix_phase)
        waveform = compute_istft(
            stft_complex,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            length=artifacts.num_samples,
            center=cfg.center,
        )
        waveform = denormalize_waveform(waveform, artifacts.waveform_norm_params)

        if cfg.validate_outputs:
            _validate_stem_output(waveform)

        waveforms[stem_name] = waveform

        if export_files and output_dir is not None:
            name_root = track_name or "separated"
            file_path = output_dir / f"{name_root}_{stem_name}.wav"
            paths[stem_name] = export_audio(waveform, file_path, artifacts.sample_rate)

    return StemOutputs(waveforms=waveforms, paths=paths, sample_rate=artifacts.sample_rate)


def separate_audio_file(
    audio_path: Union[str, Path],
    model: torch.nn.Module,
    cfg: InferenceConfig,
    stem_names: Sequence[str] = STEMS_4,
    output_dir: Union[str, Path, None] = None,
    export_files: bool = True,
) -> StemOutputs:
    """End-to-end inference: preprocess audio, run model, and reconstruct stems."""
    model_input, artifacts = prepare_model_input(audio_path, cfg)

    dev = torch.device(cfg.device)
    model_input = model_input.to(dev)

    with torch.no_grad():
        model_output = model(model_input)

    track_name = Path(audio_path).stem
    return separate_from_model_output(
        model_output=model_output,
        artifacts=artifacts,
        cfg=cfg,
        stem_names=stem_names,
        output_dir=output_dir,
        track_name=track_name,
        export_files=export_files,
    )


def separate_batch(
    audio_files: Iterable[Union[str, Path]],
    model: torch.nn.Module,
    cfg: InferenceConfig,
    stem_names: Sequence[str] = STEMS_4,
    output_dir: Union[str, Path, None] = None,
    export_files: bool = True,
) -> dict[str, StemOutputs]:
    """Run inference for multiple audio files."""
    results: dict[str, StemOutputs] = {}
    for audio_path in audio_files:
        outputs = separate_audio_file(
            audio_path=audio_path,
            model=model,
            cfg=cfg,
            stem_names=stem_names,
            output_dir=output_dir,
            export_files=export_files,
        )
        results[str(audio_path)] = outputs
    return results
