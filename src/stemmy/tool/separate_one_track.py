# src/stemmy/tool/separate_one_track.py
"""Run stem separation on a single MUSDB track directory.

This script reads TRACK_DIR/mixture.wav, runs model inference using a .pth checkpoint,
and writes separated stems to OUT_DIR.

Environment variables:
  - TRACK_DIR: Required. MUSDB track directory containing mixture.wav
  - CKPT: Optional. Path to a .pth checkpoint. If unset, uses runs/best_ckpt/unet_phase1_best.pth if present.
  - OUT_DIR: Optional. Output directory for separated stems (default: runs/separated_demo)
  - MAX_SECONDS: Optional. If > 0, truncate mixture to this duration in seconds (default: 60)
"""

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from stemmy.constants import STEMS_4
from stemmy.inference import config_from_checkpoint, load_pth_model, separate_audio_file
from stemmy.logging_config import get_logger

logger = get_logger(__name__)

STEM_NAMES: list[str] = list(STEMS_4)


class SeparationException(Exception):
    """Exception raised for separation script errors."""

    pass


class SeparateOneTrack:
    """Separates stems for a single MUSDB track directory."""

    def __init__(self) -> None:
        """Initialize the runner and resolve all required inputs from environment."""
        self.ckpt_path = self._resolve_checkpoint()
        self.track_dir = self._resolve_track_dir()
        self.out_dir = Path(os.environ.get("OUT_DIR", "runs/separated_demo")).expanduser().resolve()
        self.max_seconds = self._get_env_int("MAX_SECONDS", 60)

        logger.debug("Initialized SeparateOneTrack ckpt=%s", str(self.ckpt_path))
        logger.debug("Initialized SeparateOneTrack track_dir=%s", str(self.track_dir))
        logger.debug("Initialized SeparateOneTrack out_dir=%s", str(self.out_dir))
        logger.debug("Initialized SeparateOneTrack max_seconds=%s", str(self.max_seconds))

    def run(self) -> None:
        """Execute separation and print summary metrics."""
        mix_path = self.track_dir / "mixture.wav"
        mix, sr = self._read_mixture_stereo(mix_path, max_seconds=self.max_seconds)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, ckpt_obj = load_pth_model(str(self.ckpt_path), device=device, stems=len(STEM_NAMES))

        cfg = config_from_checkpoint(ckpt_obj)
        cfg = replace(cfg, device=device, renorm_masks=True)
        model.eval()

        if sr != cfg.sample_rate:
            raise SeparationException(
                "Sample-rate mismatch: mixture is %d, config expects %d"
                % (int(sr), int(cfg.sample_rate))
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)

        trunc_mix_path = self._maybe_write_truncated_mixture(
            mix=mix,
            sr=sr,
            out_dir=self.out_dir,
            max_seconds=self.max_seconds,
        )
        audio_path_for_infer = trunc_mix_path if trunc_mix_path is not None else mix_path

        outputs = separate_audio_file(
            audio_path=audio_path_for_infer,
            model=model,
            cfg=cfg,
            output_dir=self.out_dir,
            export_files=True,
            stems=list(STEM_NAMES),
        )

        self._finalize_output_files(outputs)

        pred_np: dict[str, np.ndarray] = {}
        for stem, waveform in outputs.stem_waveforms.items():
            if not isinstance(stem, str):
                raise SeparationException(
                    "Unexpected stem key type in stem_waveforms: %s" % type(stem).__name__
                )

            w = np.asarray(waveform)
            if w.ndim != 2:
                raise SeparationException(
                    "Unexpected waveform ndim for stem %s: got %s" % (stem, str(w.shape))
                )

            if w.shape[0] == 2:
                pred_np[stem] = w.T
            elif w.shape[1] == 2:
                pred_np[stem] = w
            else:
                raise SeparationException(
                    "Unexpected waveform shape for stem %s: got %s" % (stem, str(w.shape))
                )

        self._print_levels(pred_np)
        self._print_inner_peaks(pred_np, mix, edge=int(cfg.win_length))

        recon_snr_db = self._reconstruction_snr_db(mix, pred_np)

        seconds_used = float(mix.shape[0]) / float(sr)
        print("TRACK_DIR:", str(self.track_dir))
        print("CKPT:", str(self.ckpt_path))
        print("OUT_DIR:", str(self.out_dir))
        print("device:", device)
        print("seconds_used:", seconds_used)
        print("recon_snr_db:", recon_snr_db)

    def _finalize_output_files(self, outputs) -> None:
        """Rename exported stem files to OUT_DIR/{stem}.wav for compatibility with other tools."""
        if not hasattr(outputs, "stem_paths"):
            raise SeparationException("separate_audio_file() did not return an object with stem_paths.")
        if not hasattr(outputs, "stem_waveforms"):
            raise SeparationException(
                "separate_audio_file() did not return an object with stem_waveforms."
            )

        for stem in STEM_NAMES:
            src_path = outputs.stem_paths.get(stem)
            if src_path is None:
                raise SeparationException("Expected an exported file path for stem: %s" % stem)
            if not isinstance(src_path, Path):
                src_path = Path(src_path)

            if not src_path.is_file():
                raise SeparationException("Expected exported stem file not found: %s" % str(src_path))

            dst_path = self.out_dir / f"{stem}.wav"

            if dst_path.exists():
                if not dst_path.is_file():
                    raise SeparationException("Output path exists and is not a file: %s" % str(dst_path))
                dst_path.unlink()

            src_path.replace(dst_path)
            outputs.stem_paths[stem] = dst_path

    def _print_levels(self, pred_np: dict[str, np.ndarray]) -> None:
        """Print RMS and peak levels for each predicted stem and the mixture."""
        for name in STEM_NAMES:
            if name not in pred_np:
                raise SeparationException("Missing predicted stem for level print: %s" % name)

            x = pred_np[name]
            rms = float(np.sqrt(np.mean(x * x)))
            peak = float(np.max(np.abs(x)))
            print(f"{name:6s} rms={rms:.6f} peak={peak:.6f}")

    def _print_inner_peaks(self, pred_np: dict[str, np.ndarray], mix: np.ndarray, edge: int) -> None:
        """Print inner peak (excluding window edges) for each stem, and mixture levels."""
        if edge < 0:
            raise SeparationException("edge must be >= 0, got %d" % int(edge))

        if mix.shape[0] > 2 * edge and edge > 0:
            for name in STEM_NAMES:
                if name not in pred_np:
                    raise SeparationException("Missing predicted stem for inner-peak print: %s" % name)

                x = pred_np[name]
                inner = x[edge:-edge, :]
                inner_peak = float(np.max(np.abs(inner)))
                print(f"{name:6s} inner_peak={inner_peak:.6f}")

        mix_rms = float(np.sqrt(np.mean(mix * mix)))
        mix_peak = float(np.max(np.abs(mix)))
        print(f"mix    rms={mix_rms:.6f} peak={mix_peak:.6f}")

    def _reconstruction_snr_db(self, mix: np.ndarray, pred_np: dict[str, np.ndarray]) -> float:
        """Compute reconstruction SNR between mixture and sum of predicted stems."""
        if not STEM_NAMES:
            raise SeparationException("STEM_NAMES is empty; cannot compute reconstruction SNR.")

        stems_sum = np.zeros_like(mix)
        for stem in STEM_NAMES:
            if stem not in pred_np:
                raise SeparationException("Missing predicted stem for reconstruction: %s" % stem)
            stems_sum += pred_np[stem]

        err = mix - stems_sum
        mix_pow = float(np.mean(mix * mix) + 1e-12)
        err_pow = float(np.mean(err * err) + 1e-12)
        return float(10.0 * np.log10(mix_pow / err_pow))

    def _resolve_checkpoint(self) -> Path:
        """Resolve checkpoint path from CKPT or a repo default."""
        ckpt = self._get_env_path("CKPT")
        if ckpt is not None:
            self._validate_file(ckpt, "checkpoint (.pth)")
            return ckpt

        default_ckpt = Path("runs/best_ckpt/unet_phase1_best.pth").expanduser().resolve()
        if default_ckpt.exists():
            self._validate_file(default_ckpt, "checkpoint (.pth)")
            return default_ckpt

        raise SeparationException(
            "Set CKPT to a .pth checkpoint path, i.e. runs/best_ckpt/unet_phase1_best.pth"
        )

    def _resolve_track_dir(self) -> Path:
        """Resolve MUSDB track directory from TRACK_DIR."""
        track_dir = self._get_env_path("TRACK_DIR")
        if track_dir is None:
            raise SeparationException(
                "Set TRACK_DIR to a MUSDB track directory (contains mixture.wav and stem wavs)."
            )
        self._validate_dir(track_dir, "TRACK_DIR (MUSDB track directory)")
        return track_dir

    def _read_mixture_stereo(self, mix_path: Path, max_seconds: int) -> tuple[np.ndarray, int]:
        """Read mixture.wav as float32 stereo (N, 2), optionally truncated to max_seconds."""
        self._validate_file(mix_path, "mixture.wav")

        info = sf.info(str(mix_path))
        sr = int(info.samplerate)

        if max_seconds > 0:
            frames = int(sr * max_seconds)
            mix, read_sr = sf.read(
                str(mix_path),
                frames=frames,
                always_2d=True,
                dtype="float32",
            )
        else:
            mix, read_sr = sf.read(
                str(mix_path),
                always_2d=True,
                dtype="float32",
            )

        if int(read_sr) != sr:
            raise SeparationException(
                "soundfile returned unexpected sample rate: %d != %d" % (int(read_sr), int(sr))
            )

        if mix.ndim != 2:
            raise SeparationException("Expected mixture to be 2D (N,C), got shape %s" % (str(mix.shape),))

        if mix.shape[1] != 2:
            raise SeparationException(
                "Expected stereo mixture.wav with shape (N,2), got %s" % (str(mix.shape),)
            )

        return mix, sr

    def _maybe_write_truncated_mixture(
        self,
        mix: np.ndarray,
        sr: int,
        out_dir: Path,
        max_seconds: int,
    ) -> Optional[Path]:
        """Write a truncated mixture file when max_seconds > 0, otherwise return None."""
        if max_seconds <= 0:
            return None

        out_dir.mkdir(parents=True, exist_ok=True)

        track_name = self.track_dir.name
        trunc_path = out_dir / f"{track_name}_mixture_truncated.wav"

        sf.write(
            str(trunc_path),
            mix,
            sr,
            subtype="FLOAT",
        )
        self._validate_file(trunc_path, "truncated mixture wav")
        return trunc_path

    def _get_env_path(self, name: str) -> Optional[Path]:
        """Return an environment variable as an absolute Path, or None if unset/blank."""
        value = os.environ.get(name, "").strip()
        if value == "":
            return None
        return Path(value).expanduser().resolve()

    def _get_env_int(self, name: str, default: int) -> int:
        """Read an integer environment variable, with a safe default."""
        raw = os.environ.get(name, str(default)).strip()
        if raw == "":
            return int(default)
        try:
            return int(raw)
        except ValueError as e:
            raise SeparationException("%s must be an integer, got: %r" % (name, raw)) from e

    def _validate_file(self, path: Path, description: str) -> None:
        """Check that path exists and is a file."""
        if not path.exists():
            raise FileNotFoundError("Missing %s: %s" % (description, str(path)))
        if not path.is_file():
            raise SeparationException("Expected a file for %s: %s" % (description, str(path)))

    def _validate_dir(self, path: Path, description: str) -> None:
        """Check that path exists and is a directory."""
        if not path.exists():
            raise FileNotFoundError("Missing %s: %s" % (description, str(path)))
        if not path.is_dir():
            raise NotADirectoryError("Expected a directory for %s: %s" % (description, str(path)))


def main() -> None:
    """Entry point."""
    logger.info("Starting separate_one_track")
    runner = SeparateOneTrack()
    runner.run()
    logger.info("Finished separate_one_track")


if __name__ == "__main__":
    main()

