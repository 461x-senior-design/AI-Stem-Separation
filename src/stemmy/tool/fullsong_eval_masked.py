# src/stemmy/tool/fullsong_eval_masked.py
"""Full-song evaluation using the unified preprocessing -> inference -> postprocessing pipeline.

This script evaluates checkpoints on MUSDB18-HQ test tracks by:
- Loading a model checkpoint and its embedded configuration.
- Running separation via stemmy.inference.separate_audio_file (the authoritative stack).
- Scoring SI-SDR per stem, reconstruction SNR (mixture vs sum(pred stems)),
  and inter-stem correlation.
- Writing per-track and per-checkpoint summary CSVs.

Environment variables:
DATA:          MUSDB18-HQ root directory containing train/ and test/
CKPT_DIR:      Directory containing .pth checkpoints
EVAL_DIR:      Output directory for CSV results
DEVICE:        Torch device string (default: "cuda")
N_EVAL_TRACKS: Number of test tracks to evaluate (default: "30")
MAX_SECONDS:   Max seconds per track to evaluate (default: "30", 0 = full track)
"""

import csv
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from stemmy.constants import STEMS_4
from stemmy.inference import (
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    separate_audio_file,
)
from stemmy.logging_config import get_logger

logger = get_logger(__name__)

STEMS: list[str] = list(STEMS_4)


class EvaluationException(Exception):
    """Exception raised for evaluation script errors."""

    pass


def _get_env_path(name: str) -> Path:
    """Read a required environment variable as a Path."""
    val = os.environ.get(name, "").strip()
    if val == "":
        raise EvaluationException("Missing required environment variable: %s" % name)
    return Path(val).expanduser().resolve()


def _get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable, or return default when unset/blank."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise EvaluationException(
            "Environment variable %s must be an int, got: %s" % (name, raw)
        ) from exc


def _validate_device(device: str) -> str:
    """Validate and normalize a torch device string."""
    dev = (device or "").strip()
    if dev == "":
        dev = "cuda"

    if dev == "cpu":
        return dev

    if dev == "cuda" or dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise EvaluationException(
                "Requested CUDA device but torch.cuda.is_available() is False."
            )
        return dev

    raise EvaluationException("Invalid DEVICE. Use cpu, cuda, or cuda:N (example: cuda:0).")


def _list_tracks(split_dir: Path) -> list[Path]:
    """Return MUSDB track directories that contain mixture.wav and all required stems."""
    tracks: list[Path] = []
    if not split_dir.is_dir():
        return tracks

    for track_dir in sorted(split_dir.iterdir()):
        if not track_dir.is_dir():
            continue

        mix_path = track_dir / "mixture.wav"
        if not mix_path.is_file():
            continue

        ok = True
        for stem in STEMS:
            if not (track_dir / ("%s.wav" % stem)).is_file():
                ok = False
                break

        if ok:
            tracks.append(track_dir)

    return tracks


def _read_slice(path: Path, max_samples: int) -> tuple[np.ndarray, int]:
    """Read audio as float32 stereo (N, 2) and optionally truncate to max_samples."""
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if x.ndim != 2 or x.shape[1] != 2:
        raise EvaluationException(
            "Expected stereo audio with shape (N,2), got: %s" % (str(x.shape),)
        )
    if max_samples > 0 and x.shape[0] > max_samples:
        x = x[:max_samples, :]
    return x, int(sr)


def _zero_mean(x: np.ndarray) -> np.ndarray:
    """Remove per-channel DC offset."""
    if x.ndim != 2:
        raise EvaluationException("Expected 2D array for stereo audio, got: %s" % (str(x.shape),))
    return x - np.mean(x, axis=0, keepdims=True)


def _si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    """Compute SI-SDR for stereo signals by averaging channel-wise SI-SDR."""
    est_f = est.astype(np.float64, copy=False)
    ref_f = ref.astype(np.float64, copy=False)

    if est_f.shape != ref_f.shape:
        raise EvaluationException(
            "Shape mismatch est=%s ref=%s" % (str(est_f.shape), str(ref_f.shape))
        )
    if est_f.ndim != 2 or est_f.shape[1] != 2:
        raise EvaluationException("Expected stereo shape (N,2), got: %s" % (str(est_f.shape),))

    est_zm = _zero_mean(est_f)
    ref_zm = _zero_mean(ref_f)

    sdrs: list[float] = []
    for c in range(ref_zm.shape[1]):
        r = ref_zm[:, c]
        e = est_zm[:, c]

        rr = float(np.dot(r, r) + eps)
        a = float(np.dot(e, r) / rr)

        s_target = a * r
        e_noise = e - s_target

        num = float(np.dot(s_target, s_target) + eps)
        den = float(np.dot(e_noise, e_noise) + eps)
        sdrs.append(10.0 * math.log10(num / den))

    return float(np.mean(sdrs))


def _recon_snr_db(mix: np.ndarray, stems_sum: np.ndarray, eps: float = 1e-12) -> float:
    """Compute reconstruction SNR between mixture and sum of predicted stems."""
    mix_f = mix.astype(np.float64, copy=False)
    stems_f = stems_sum.astype(np.float64, copy=False)

    if mix_f.shape != stems_f.shape:
        raise EvaluationException(
            "Shape mismatch mix=%s stems_sum=%s" % (str(mix_f.shape), str(stems_f.shape))
        )

    err = mix_f - stems_f
    mix_pow = float(np.mean(mix_f * mix_f) + eps)
    err_pow = float(np.mean(err * err) + eps)
    return float(10.0 * math.log10(mix_pow / err_pow))


def _corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute a simple normalized correlation between two stereo signals."""
    if a.shape != b.shape:
        raise EvaluationException(
            "Shape mismatch for correlation: a=%s b=%s" % (str(a.shape), str(b.shape))
        )

    a_f = a.reshape(-1).astype(np.float64, copy=False)
    b_f = b.reshape(-1).astype(np.float64, copy=False)

    a_f = a_f - float(a_f.mean())
    b_f = b_f - float(b_f.mean())

    den = (np.sqrt(np.mean(a_f * a_f)) * np.sqrt(np.mean(b_f * b_f))) + eps
    return float(np.mean(a_f * b_f) / den)


def _mean_interstem_corr(pred_stems: dict[str, np.ndarray]) -> float:
    """Compute mean pairwise correlation across all predicted stems."""
    vals: list[float] = []
    for i in range(len(STEMS)):
        for j in range(i + 1, len(STEMS)):
            vals.append(_corr(pred_stems[STEMS[i]], pred_stems[STEMS[j]]))
    return float(np.mean(vals)) if vals else 0.0


def _ckpt_sort_key(p: Path) -> tuple[int, int, str]:
    """Sort checkpoints to prefer those with an epoch number, then ascending epoch."""
    stem = p.stem
    e = -1

    if "epoch" in stem:
        tail = stem.split("epoch")[-1]
        try:
            e = int(tail)
        except ValueError:
            e = -1

    unknown = 1 if e < 0 else 0
    epoch_val = e if e >= 0 else 0
    return unknown, epoch_val, p.name


def _truncate_outputs_to_mix_len(
    pred: dict[str, np.ndarray], mix_len: int
) -> dict[str, np.ndarray]:
    """Trim or pad predicted stems to match the mixture length (N)."""
    if mix_len < 0:
        raise EvaluationException("mix_len must be >= 0, got %d" % int(mix_len))

    out: dict[str, np.ndarray] = {}
    for stem in STEMS:
        if stem not in pred:
            raise EvaluationException("Missing predicted stem: %s" % stem)

        x = pred[stem]
        if x.ndim != 2 or x.shape[1] != 2:
            raise EvaluationException(
                "Expected pred stem %s to have shape (N,2), got %s" % (stem, str(x.shape))
            )

        if x.shape[0] > mix_len:
            out[stem] = x[:mix_len, :]
        elif x.shape[0] < mix_len:
            pad = np.zeros((mix_len - x.shape[0], x.shape[1]), dtype=x.dtype)
            out[stem] = np.concatenate([x, pad], axis=0)
        else:
            out[stem] = x

    return out


def _maybe_truncate_mixture_wav(
    mix_path: Path,
    eval_dir: Path,
    mix: np.ndarray,
    sr: int,
    max_seconds: int,
) -> Path:
    """Write a truncated mixture.wav when MAX_SECONDS > 0 so inference matches evaluation slice."""
    if max_seconds <= 0:
        return mix_path

    trunc_dir = eval_dir / "_truncated_mixtures"
    trunc_dir.mkdir(parents=True, exist_ok=True)

    trunc_path = trunc_dir / ("%s_trunc_%ds.wav" % (mix_path.parent.name, int(max_seconds)))
    if not trunc_path.exists():
        sf.write(str(trunc_path), mix, sr, subtype="FLOAT")

    return trunc_path


def _build_eval_inference_config(
    ckpt_obj: dict,
    device: str,
) -> InferenceConfig:
    """Build an InferenceConfig for evaluation based on checkpoint config.

    Evaluation overrides:
    - stems ordering is fixed to project canonical STEMS_4 ordering
    - export_files=False (evaluation writes only CSVs)
    - renorm_masks=True (keeps predicted masks sum-to-one across stems per TF-bin)
    """
    cfg_from_ckpt = config_from_checkpoint(ckpt_obj)
    cfg = replace(
        cfg_from_ckpt,
        stems=list(STEMS),
        device=device,
        export_files=False,
        renorm_masks=True,
    )
    return cfg


def main() -> None:
    """Entry point for full-song evaluation."""
    data_root = _get_env_path("DATA")
    ckpt_dir = _get_env_path("CKPT_DIR")
    eval_dir = _get_env_path("EVAL_DIR")

    device_raw = os.environ.get("DEVICE", "cuda")
    device = _validate_device(device_raw)

    n_eval_tracks = _get_env_int("N_EVAL_TRACKS", 30)
    max_seconds = _get_env_int("MAX_SECONDS", 30)

    if n_eval_tracks <= 0:
        raise EvaluationException("N_EVAL_TRACKS must be > 0, got %d" % int(n_eval_tracks))
    if max_seconds < 0:
        raise EvaluationException("MAX_SECONDS must be >= 0, got %d" % int(max_seconds))

    eval_dir.mkdir(parents=True, exist_ok=True)

    tracks = _list_tracks(data_root / "test")[:n_eval_tracks]
    if not tracks:
        raise EvaluationException("No valid test tracks found in: %s" % str(data_root / "test"))

    ckpts = sorted(ckpt_dir.glob("*.pth"), key=_ckpt_sort_key)
    if not ckpts:
        raise EvaluationException("No checkpoints found in: %s" % str(ckpt_dir))

    per_track_csv = eval_dir / "fullsong_eval_per_track.csv"
    summary_csv = eval_dir / "fullsong_eval_summary.csv"

    with per_track_csv.open("w", newline="") as f_track, summary_csv.open("w", newline="") as f_sum:
        track_writer = csv.writer(f_track)
        sum_writer = csv.writer(f_sum)

        track_header = ["ckpt", "track"]
        for stem in STEMS:
            track_header.append("sisdr_%s" % stem)
        track_header += ["recon_snr_db", "interstem_corr"]
        track_writer.writerow(track_header)

        sum_header = ["ckpt"]
        for stem in STEMS:
            sum_header.append("mean_sisdr_%s" % stem)
        sum_header += ["mean_recon_snr_db", "mean_interstem_corr"]
        sum_writer.writerow(sum_header)

        for ckpt_path in ckpts:
            logger.info("Evaluating checkpoint: %s", str(ckpt_path))

            model, ckpt_obj = load_pth_model(str(ckpt_path), device=device, stems=len(STEMS))
            model.eval()

            cfg = _build_eval_inference_config(ckpt_obj, device=device)

            sisdr_accum: dict[str, list[float]] = {s: [] for s in STEMS}
            recon_accum: list[float] = []
            corr_accum: list[float] = []

            for track_dir in tracks:
                track_name = track_dir.name
                mix_path = track_dir / "mixture.wav"

                max_samples = 0 if max_seconds <= 0 else int(max_seconds * int(cfg.sample_rate))
                mix, sr = _read_slice(mix_path, max_samples=max_samples)

                if sr != int(cfg.sample_rate):
                    raise EvaluationException(
                        "Track sample rate mismatch: %s sr=%d expected=%d"
                        % (str(mix_path), int(sr), int(cfg.sample_rate))
                    )

                infer_mix_path = _maybe_truncate_mixture_wav(
                    mix_path=mix_path,
                    eval_dir=eval_dir,
                    mix=mix,
                    sr=sr,
                    max_seconds=max_seconds,
                )

                out = separate_audio_file(
                    audio_path=infer_mix_path,
                    model=model,
                    cfg=cfg,
                    output_dir=eval_dir,
                    export_files=False,
                    stems=list(STEMS),
                    checkpoint=ckpt_obj,
                )

                pred_stems: dict[str, np.ndarray] = {}
                for stem in STEMS:
                    wav = out.stem_waveforms[stem]
                    w = np.asarray(wav)
                    if w.ndim != 2:
                        raise EvaluationException(
                            "Pred stem has unexpected ndim for %s: %s" % (stem, str(w.shape))
                        )

                    # Postprocessor returns [2, N]. Standardize to (N, 2) for scoring.
                    if w.shape[0] == 2:
                        pred_stems[stem] = w.T.astype(np.float32, copy=False)
                    elif w.shape[1] == 2:
                        pred_stems[stem] = w.astype(np.float32, copy=False)
                    else:
                        raise EvaluationException(
                            "Pred stem has unexpected shape for %s: %s" % (stem, str(w.shape))
                        )

                pred_stems = _truncate_outputs_to_mix_len(pred_stems, mix.shape[0])

                gt: dict[str, np.ndarray] = {}
                for stem in STEMS:
                    gt_path = track_dir / ("%s.wav" % stem)
                    gt_x, gt_sr = _read_slice(gt_path, max_samples=max_samples)

                    if gt_sr != int(cfg.sample_rate):
                        raise EvaluationException(
                            "GT sample rate mismatch: %s sr=%d expected=%d"
                            % (str(gt_path), int(gt_sr), int(cfg.sample_rate))
                        )

                    # Align GT lengths to mixture length.
                    if gt_x.shape[0] != mix.shape[0]:
                        if gt_x.shape[0] > mix.shape[0]:
                            gt_x = gt_x[: mix.shape[0], :]
                        else:
                            pad = np.zeros(
                                (mix.shape[0] - gt_x.shape[0], gt_x.shape[1]),
                                dtype=gt_x.dtype,
                            )
                            gt_x = np.concatenate([gt_x, pad], axis=0)

                    gt[stem] = gt_x

                sisdr_row: dict[str, float] = {}
                for stem in STEMS:
                    v = _si_sdr(pred_stems[stem], gt[stem])
                    sisdr_row[stem] = v
                    sisdr_accum[stem].append(v)

                stems_sum = np.zeros_like(mix)
                for stem in STEMS:
                    stems_sum += pred_stems[stem]
                recon = _recon_snr_db(mix, stems_sum)
                recon_accum.append(recon)

                corr_val = _mean_interstem_corr(pred_stems)
                corr_accum.append(corr_val)

                track_row = [str(ckpt_path), track_name]
                for stem in STEMS:
                    track_row.append("%.6f" % sisdr_row[stem])
                track_row.append("%.6f" % recon)
                track_row.append("%.6f" % corr_val)
                track_writer.writerow(track_row)

            mean_row = [str(ckpt_path)]
            for stem in STEMS:
                mean_row.append(
                    "%.6f" % float(np.mean(sisdr_accum[stem]) if sisdr_accum[stem] else 0.0)
                )
            mean_row.append("%.6f" % float(np.mean(recon_accum) if recon_accum else 0.0))
            mean_row.append("%.6f" % float(np.mean(corr_accum) if corr_accum else 0.0))
            sum_writer.writerow(mean_row)

    logger.info("Wrote per-track CSV: %s", str(per_track_csv))
    logger.info("Wrote summary CSV: %s", str(summary_csv))


if __name__ == "__main__":
    raise SystemExit(main())
