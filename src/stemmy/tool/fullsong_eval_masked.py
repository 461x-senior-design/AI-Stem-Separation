# src/stemmy/tool/fullsong_eval_masked.py
"""Full-song evaluation using the preprocessing, inference, and postprocessing pipeline.

This script:
- Loads a model checkpoint and its embedded configuration.
- Runs separation on held-out MUSDB18-HQ tracks.
- Scores SI-SDR per stem, reconstruction SNR, and inter-stem correlation.
- Writes per-track and per-checkpoint summary CSVs.

Environment variables:
DATA:          MUSDB18-HQ root directory containing train/ and test/
CKPT_DIR:      Directory containing .pth checkpoints
EVAL_DIR:      Output directory for CSV results
DEVICE:        Torch device string (default: "cuda")
N_EVAL_TRACKS: Number of tracks to evaluate (default: "30")
MAX_SECONDS:   Max seconds per track to evaluate (default: "30", 0 = full track)
EVAL_SUBSET:   MUSDB subset to evaluate on (default: "test")

Optional environment variables:
EVAL_PROGRESS:            1 to print checkpoint/track progress to stdout (default: "1")
EVAL_PRINT_METRICS:       1 to print per-track metrics lines (default: "0")
EVAL_FLUSH_EVERY:         Flush CSV files every N per-track rows (default: "1")
EVAL_FSYNC_EVERY:         fsync CSV files every N per-track rows (0 disables, default: "0")
EVAL_PRINT_EVERY_TRACKS:  Print a progress line every N tracks (default: "1")
EVAL_EVERY_N_CKPTS:       Evaluate only every Nth checkpoint by epoch order (default: "1" = all)
EVAL_CHUNK_FRAMES:        Inference chunk size in STFT frames (default: "256", 0 disables)
EVAL_OVERLAP_FRAMES:      Inference chunk overlap in STFT frames (default: "64")
EVAL_AMP:                 1 to use CUDA autocast during evaluation inference (default: "1")

Watch mode (parallel eval during training):
EVAL_WATCH_MODE:          1 to watch CKPT_DIR for new checkpoints and evaluate as they appear.
TRAIN_DONE_FILE:          Path to a flag file; when it exists, watcher does a final scan and exits.
EVAL_WATCH_POLL_SECONDS:  How often (seconds) to poll for new checkpoints (default: "60").
"""

import csv
import math
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
import wandb
from rich.console import Console
from rich.table import Table

from stemmy.constants import (
    BOLD_PURPLE,
    LAVENDER,
    NEON_GREEN,
    ROSE_RED,
    STEMS_4,
    TARGET_CHANNELS,
    WHITE,
)
from stemmy.inference import (
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    separate_audio_file,
)
from stemmy.logging_config import get_logger, setup_logging
from stemmy.tool.progress_theme import create_themed_progress, start_eq_animator
from stemmy.wandb_config import wandb_run

logger = get_logger(__name__)

STEMS: list[str] = list(STEMS_4)
SUPPORTED_EVAL_SUBSETS: tuple[str, ...] = ("test",)

# Running "best checkpoint so far" per wandb run id, populated by
# _eval_one_checkpoint.
_STEMMY_BEST_BY_RUN: dict[str, dict] = {}


class EvaluationException(Exception):
    """Exception raised for evaluation script errors."""

    pass


def _coerce_epoch(value: object, fallback: int) -> int:
    """Return a plain int epoch for CSV/W&B logging."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(fallback)


def _publish_wandb_metrics(
    metrics: dict[str, float], step: int, timeout_seconds: float = 10.0
) -> None:
    """Publish W&B metrics without allowing a stuck network call to stall evaluation."""
    run = wandb.run
    if run is None:
        return

    error: list[BaseException] = []

    def _worker() -> None:
        try:
            run.log(metrics, step=int(step), commit=True)
            run.summary.update(metrics)
        except BaseException as exc:  # pragma: no cover - defensive around external service
            error.append(exc)

    thread = threading.Thread(target=_worker, name="wandb-eval-log", daemon=True)
    thread.start()
    thread.join(float(timeout_seconds))

    if thread.is_alive():
        logger.warning(
            "Timed out after %.1fs while logging eval metrics to W&B run %s; "
            "continuing because CSV metrics were already written.",
            float(timeout_seconds),
            getattr(run, "id", "unknown"),
        )
        return

    if error:
        logger.warning("Failed to log eval metrics to W&B: %s", error[0])
        return

    logger.info(
        "Logged eval metrics to W&B run %s at checkpoint epoch %d.",
        getattr(run, "id", "unknown"),
        int(step),
    )


def _get_env_path(name: str) -> Path:
    """Read a required environment variable as a Path."""
    val = os.environ.get(name, "").strip()
    if val == "":
        raise EvaluationException("Missing required environment variable: %s" % name)
    return Path(val).expanduser().resolve()


def _get_env_str(name: str, default: str) -> str:
    """Read a string environment variable, or return default when unset or blank."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return str(default)
    return str(raw)


def _get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable, or return default when unset or blank."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise EvaluationException(
            "Environment variable %s must be an int, got: %s" % (name, raw)
        ) from exc


def _get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean-like environment variable."""
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return bool(default)

    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}

    if raw in truthy:
        return True
    if raw in falsy:
        return False

    raise EvaluationException(
        "Environment variable %s must be bool-like (1/0/true/false/yes/no), got: %s" % (name, raw)
    )


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


def _validate_eval_subset(eval_subset: str) -> str:
    """Validate the MUSDB subset used for evaluation."""
    subset = str(eval_subset).strip().lower()
    if subset not in SUPPORTED_EVAL_SUBSETS:
        raise EvaluationException(
            "Unsupported EVAL_SUBSET '%s'. Expected one of %s." % (subset, SUPPORTED_EVAL_SUBSETS)
        )
    return subset


def _validate_eval_inference_options(
    eval_chunk_frames: int,
    eval_overlap_frames: int,
) -> None:
    """Validate evaluation inference chunking options."""
    if eval_chunk_frames < 0:
        raise EvaluationException("EVAL_CHUNK_FRAMES must be >= 0, got %d" % int(eval_chunk_frames))
    if eval_overlap_frames < 0:
        raise EvaluationException(
            "EVAL_OVERLAP_FRAMES must be >= 0, got %d" % int(eval_overlap_frames)
        )
    if eval_chunk_frames == 0 and eval_overlap_frames != 0:
        raise EvaluationException("EVAL_OVERLAP_FRAMES must be 0 when EVAL_CHUNK_FRAMES is 0.")
    if eval_chunk_frames > 0 and eval_overlap_frames >= eval_chunk_frames:
        raise EvaluationException(
            "EVAL_OVERLAP_FRAMES must be < EVAL_CHUNK_FRAMES when chunking is enabled."
        )


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
    if x.ndim != 2 or x.shape[1] != TARGET_CHANNELS:
        raise EvaluationException(
            "Expected stereo audio with shape (N,%d), got: %s"
            % (int(TARGET_CHANNELS), str(x.shape))
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
    if est_f.ndim != 2 or est_f.shape[1] != TARGET_CHANNELS:
        raise EvaluationException(
            "Expected stereo shape (N,%d), got: %s" % (int(TARGET_CHANNELS), str(est_f.shape))
        )

    est_zm = _zero_mean(est_f)
    ref_zm = _zero_mean(ref_f)

    sdrs: list[float] = []
    for channel in range(ref_zm.shape[1]):
        ref_channel = ref_zm[:, channel]
        est_channel = est_zm[:, channel]

        ref_energy = float(np.dot(ref_channel, ref_channel) + eps)
        scale = float(np.dot(est_channel, ref_channel) / ref_energy)

        target_component = scale * ref_channel
        noise_component = est_channel - target_component

        numerator = float(np.dot(target_component, target_component) + eps)
        denominator = float(np.dot(noise_component, noise_component) + eps)
        sdrs.append(10.0 * math.log10(numerator / denominator))

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
    """Compute a normalized correlation between two stereo signals."""
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
    """Compute mean pairwise correlation across predicted stems."""
    vals: list[float] = []
    for i in range(len(STEMS)):
        for j in range(i + 1, len(STEMS)):
            vals.append(_corr(pred_stems[STEMS[i]], pred_stems[STEMS[j]]))
    return float(np.mean(vals)) if vals else 0.0


def _ckpt_sort_key(p: Path) -> tuple[int, int, str]:
    """Sort checkpoints to prefer those with an epoch number, then ascending epoch."""
    stem = p.stem
    epoch = -1

    if "epoch" in stem:
        tail = stem.split("epoch")[-1]
        try:
            epoch = int(tail)
        except ValueError:
            epoch = -1

    unknown = 1 if epoch < 0 else 0
    epoch_val = epoch if epoch >= 0 else 0
    return unknown, epoch_val, p.name


def _truncate_outputs_to_mix_len(
    pred: dict[str, np.ndarray],
    mix_len: int,
) -> dict[str, np.ndarray]:
    """Trim or pad predicted stems to match the mixture length."""
    if mix_len < 0:
        raise EvaluationException("mix_len must be >= 0, got %d" % int(mix_len))

    out: dict[str, np.ndarray] = {}
    for stem in STEMS:
        if stem not in pred:
            raise EvaluationException("Missing predicted stem: %s" % stem)

        x = pred[stem]
        if x.ndim != 2 or x.shape[1] != TARGET_CHANNELS:
            raise EvaluationException(
                "Expected pred stem %s to have shape (N,%d), got %s"
                % (stem, int(TARGET_CHANNELS), str(x.shape))
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
    """Write a truncated mixture.wav when MAX_SECONDS > 0 so inference matches the slice."""
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
    eval_chunk_frames: int,
    eval_overlap_frames: int,
    eval_amp: bool,
) -> InferenceConfig:
    """Build an InferenceConfig for evaluation based on checkpoint config."""
    cfg_from_ckpt = config_from_checkpoint(ckpt_obj)
    cfg = replace(
        cfg_from_ckpt,
        stems=list(STEMS),
        device=device,
        export_files=False,
        renorm_masks=True,
        chunk_frames=int(eval_chunk_frames),
        overlap_frames=int(eval_overlap_frames),
        amp=bool(eval_amp),
    )
    return cfg


def _print_progress(enabled: bool, msg: str) -> None:
    """Print progress to stdout with immediate flush."""
    if not enabled:
        return
    print(msg, flush=True)


def _flush_outputs(
    f_track,
    f_sum,
    do_fsync: bool,
) -> None:
    """Flush CSV file buffers and optionally fsync."""
    f_track.flush()
    f_sum.flush()
    if do_fsync:
        os.fsync(f_track.fileno())
        os.fsync(f_sum.fileno())


def _vocals_priority_score(
    mean_sisdr_by_stem: dict[str, float],
    mean_recon: float,
    mean_corr: float,
) -> float:
    """Compute a derived score that weights vocals more heavily."""
    mean_sisdr = float(np.mean([mean_sisdr_by_stem[stem] for stem in STEMS]))
    vocals = float(mean_sisdr_by_stem.get("vocals", 0.0))
    return mean_sisdr + 0.5 * vocals + 0.05 * mean_recon + 2.0 * mean_corr


def print_evaluation_summary(rows: list[dict[str, Union[float, str]]]) -> None:
    """Print an end-of-run checkpoint evaluation summary table."""
    if not rows:
        return

    console = Console()
    table = Table(title="Evaluation Summary", header_style=BOLD_PURPLE)
    table.add_column("Checkpoint", style=LAVENDER)
    table.add_column("Mean SI-SDR", justify="right", style=WHITE)
    for stem in STEMS:
        table.add_column("SI-SDR %s" % stem, justify="right", style=WHITE)
    table.add_column("Recon SNR", justify="right", style=WHITE)
    table.add_column("Interstem Corr", justify="right", style=WHITE)
    table.add_column("Vocals Priority", justify="right", style=WHITE)

    score_keys = [
        "mean_sisdr",
        *["mean_sisdr_%s" % stem for stem in STEMS],
        "mean_recon_snr_db",
        "mean_interstem_corr",
        "vocals_priority_score",
    ]
    col_best: dict[str, float] = {}
    col_worst: dict[str, float] = {}
    for key in score_keys:
        values = [float(row[key]) for row in rows]
        col_best[key] = max(values)
        col_worst[key] = min(values)

    def style_score(key: str, value: float, decimals: int = 3) -> str:
        if value == col_best[key]:
            return f"[{NEON_GREEN}]{value:.{decimals}f}[/]"
        if value == col_worst[key]:
            return f"[{ROSE_RED}]{value:.{decimals}f}[/]"
        return f"[{WHITE}]{value:.{decimals}f}[/]"

    for row in rows:
        mean_sisdr = float(row["mean_sisdr"])
        mean_recon = float(row["mean_recon_snr_db"])
        mean_corr = float(row["mean_interstem_corr"])
        vocals_priority = float(row["vocals_priority_score"])
        table.add_row(
            Path(str(row["ckpt"])).name,
            style_score("mean_sisdr", mean_sisdr, decimals=3),
            *[
                style_score("mean_sisdr_%s" % stem, float(row["mean_sisdr_%s" % stem]), decimals=3)
                for stem in STEMS
            ],
            style_score("mean_recon_snr_db", mean_recon, decimals=3),
            style_score("mean_interstem_corr", mean_corr, decimals=4),
            style_score("vocals_priority_score", vocals_priority, decimals=3),
        )

    console.print()
    console.print(table)


def _eval_one_checkpoint(
    ckpt_path: Path,
    ckpt_idx: int,
    total_ckpts: int,
    tracks: list[Path],
    device: str,
    eval_chunk_frames: int,
    eval_overlap_frames: int,
    eval_amp: bool,
    max_seconds: int,
    eval_dir: Path,
    track_writer,
    sum_writer,
    f_track,
    f_sum,
    flush_every: int,
    fsync_every: int,
    per_track_rows_written: int,
    text_progress_enabled: bool,
    print_metrics: bool,
    print_every_tracks: int,
    progress,
    ckpt_task_id,
    track_task_id,
) -> tuple[dict, int]:
    """Evaluate one checkpoint over all tracks."""
    total_ckpts_str = str(total_ckpts) if total_ckpts >= 0 else "?"
    ckpt_label = str(ckpt_path)
    total_tracks = len(tracks)
    eval_t_start = time.time()

    logger.info("Evaluating checkpoint: %s", ckpt_label)
    _print_progress(
        text_progress_enabled,
        "--- ckpt %d/%s: %s ---" % (int(ckpt_idx), total_ckpts_str, ckpt_label),
    )

    model, ckpt_obj = load_pth_model(
        str(ckpt_path),
        device=device,
        stems=len(STEMS),
    )
    model.eval()

    cfg = _build_eval_inference_config(
        ckpt_obj=ckpt_obj,
        device=device,
        eval_chunk_frames=eval_chunk_frames,
        eval_overlap_frames=eval_overlap_frames,
        eval_amp=eval_amp,
    )

    sisdr_accum: dict[str, list[float]] = {s: [] for s in STEMS}
    recon_accum: list[float] = []
    corr_accum: list[float] = []

    max_samples = 0 if max_seconds <= 0 else int(max_seconds * int(cfg.sample_rate))

    for track_idx, track_dir in enumerate(tracks, start=1):
        if (int(track_idx) == 1) or ((int(track_idx) % int(print_every_tracks)) == 0):
            _print_progress(
                text_progress_enabled,
                "ckpt %d/%s  track %d/%d  %s"
                % (
                    int(ckpt_idx),
                    total_ckpts_str,
                    int(track_idx),
                    int(total_tracks),
                    track_dir.name,
                ),
            )

        track_name = track_dir.name
        mix_path = track_dir / "mixture.wav"

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

            if w.shape[0] == TARGET_CHANNELS:
                pred_stems[stem] = w.T.astype(np.float32, copy=False)
            elif w.shape[1] == TARGET_CHANNELS:
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
            sisdr_value = _si_sdr(pred_stems[stem], gt[stem])
            sisdr_row[stem] = sisdr_value
            sisdr_accum[stem].append(sisdr_value)

        stems_sum = np.zeros_like(mix)
        for stem in STEMS:
            stems_sum += pred_stems[stem]
        recon = _recon_snr_db(mix, stems_sum)
        recon_accum.append(recon)

        corr_val = _mean_interstem_corr(pred_stems)
        corr_accum.append(corr_val)

        track_row = [ckpt_label, track_name]
        for stem in STEMS:
            track_row.append("%.6f" % float(sisdr_row[stem]))
        track_row.append("%.6f" % float(recon))
        track_row.append("%.6f" % float(corr_val))
        track_writer.writerow(track_row)

        per_track_rows_written += 1

        do_flush = (int(per_track_rows_written) % int(flush_every)) == 0
        do_fsync = (int(fsync_every) > 0) and (
            (int(per_track_rows_written) % int(fsync_every)) == 0
        )
        if do_flush or do_fsync:
            _flush_outputs(f_track, f_sum, do_fsync=bool(do_fsync))

        if progress is not None and track_task_id is not None:
            progress.update(
                track_task_id,
                advance=1,
                description="ckpt %d/%s  %s" % (int(ckpt_idx), total_ckpts_str, track_name),
            )

    mean_sisdr_by_stem = {
        stem: float(np.mean(sisdr_accum[stem]) if sisdr_accum[stem] else 0.0) for stem in STEMS
    }
    mean_recon = float(np.mean(recon_accum) if recon_accum else 0.0)
    mean_corr = float(np.mean(corr_accum) if corr_accum else 0.0)
    mean_sisdr = float(np.mean([mean_sisdr_by_stem[stem] for stem in STEMS]))
    vocals_priority_score = _vocals_priority_score(
        mean_sisdr_by_stem=mean_sisdr_by_stem,
        mean_recon=mean_recon,
        mean_corr=mean_corr,
    )

    mean_row = [ckpt_label, "%.6f" % float(mean_sisdr)]
    for stem in STEMS:
        mean_row.append("%.6f" % float(mean_sisdr_by_stem[stem]))
    mean_row.append("%.6f" % float(mean_recon))
    mean_row.append("%.6f" % float(mean_corr))
    mean_row.append("%.6f" % float(vocals_priority_score))
    sum_writer.writerow(mean_row)

    _flush_outputs(
        f_track,
        f_sum,
        do_fsync=bool(
            (int(fsync_every) > 0) and ((int(per_track_rows_written) % int(fsync_every)) == 0)
        ),
    )

    eval_elapsed = time.time() - eval_t_start

    if wandb.run is not None:
        ckpt_epoch = _coerce_epoch(ckpt_obj.get("epoch", ckpt_idx), ckpt_idx)
        metrics: dict[str, float] = {}
        for stem in STEMS:
            vals = sisdr_accum[stem]
            metrics["eval/sisdr_%s_mean" % stem] = mean_sisdr_by_stem[stem]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                metrics["eval/sisdr_%s_median" % stem] = float(np.median(arr))
        metrics["eval/mean_sisdr"] = mean_sisdr
        metrics["eval/mean_recon_snr_db"] = mean_recon
        metrics["eval/mean_interstem_corr"] = mean_corr
        metrics["eval/vocals_priority_score"] = vocals_priority_score
        metrics["eval/checkpoint_eval_seconds"] = eval_elapsed
        metrics["checkpoint/epoch"] = float(ckpt_epoch)

        # Running best across this watcher's lifetime.
        run_key = str(getattr(wandb.run, "id", "") or id(wandb.run))
        best = _STEMMY_BEST_BY_RUN.get(run_key)
        if best is None or mean_sisdr > best["sisdr"]:
            best = {"sisdr": mean_sisdr, "epoch": int(ckpt_epoch)}
            _STEMMY_BEST_BY_RUN[run_key] = best
        metrics["eval/best_sisdr_so_far"] = best["sisdr"]
        metrics["eval/best_epoch_so_far"] = best["epoch"]

        logger.info(
            "Publishing eval metrics for %s to W&B run %s: epoch=%d mean_sisdr=%.6f",
            Path(ckpt_label).name,
            getattr(wandb.run, "id", "unknown"),
            int(ckpt_epoch),
            float(mean_sisdr),
        )
        _publish_wandb_metrics(metrics, step=int(ckpt_epoch))

    if progress is not None and ckpt_task_id is not None:
        progress.update(
            ckpt_task_id,
            advance=1,
            description=Path(ckpt_label).name,
        )

    summary_row: dict[str, Union[float, str]] = {
        "ckpt": ckpt_label,
        "mean_sisdr": mean_sisdr,
        "mean_recon_snr_db": mean_recon,
        "mean_interstem_corr": mean_corr,
        "vocals_priority_score": vocals_priority_score,
        **{f"mean_sisdr_{stem}": mean_sisdr_by_stem[stem] for stem in STEMS},
    }
    return summary_row, per_track_rows_written


@wandb_run(job_type="evaluation", name="song_eval")
def main() -> None:
    """Run full-song evaluation."""
    setup_logging()

    progress_enabled = _get_env_bool("EVAL_PROGRESS", True)
    print_metrics = _get_env_bool("EVAL_PRINT_METRICS", False)

    flush_every = _get_env_int("EVAL_FLUSH_EVERY", 1)
    if flush_every <= 0:
        raise EvaluationException("EVAL_FLUSH_EVERY must be > 0, got %d" % int(flush_every))

    fsync_every = _get_env_int("EVAL_FSYNC_EVERY", 0)
    if fsync_every < 0:
        raise EvaluationException("EVAL_FSYNC_EVERY must be >= 0, got %d" % int(fsync_every))

    print_every_tracks = _get_env_int("EVAL_PRINT_EVERY_TRACKS", 1)
    if print_every_tracks <= 0:
        raise EvaluationException(
            "EVAL_PRINT_EVERY_TRACKS must be > 0, got %d" % int(print_every_tracks)
        )

    data_root = _get_env_path("DATA")
    ckpt_dir = _get_env_path("CKPT_DIR")
    eval_dir = _get_env_path("EVAL_DIR")

    device_raw = os.environ.get("DEVICE", "cuda")
    device = _validate_device(device_raw)

    n_eval_tracks = _get_env_int("N_EVAL_TRACKS", 30)
    max_seconds = _get_env_int("MAX_SECONDS", 30)
    eval_subset = _validate_eval_subset(_get_env_str("EVAL_SUBSET", "test"))

    eval_every_n_ckpts = _get_env_int("EVAL_EVERY_N_CKPTS", 1)
    eval_chunk_frames = _get_env_int("EVAL_CHUNK_FRAMES", 256)
    eval_overlap_frames = _get_env_int("EVAL_OVERLAP_FRAMES", 64)
    eval_amp = _get_env_bool("EVAL_AMP", True)
    _validate_eval_inference_options(
        eval_chunk_frames=eval_chunk_frames,
        eval_overlap_frames=eval_overlap_frames,
    )

    # Watch mode: poll CKPT_DIR for new checkpoints while training runs.
    watch_mode = _get_env_bool("EVAL_WATCH_MODE", False)
    train_done_file_str = os.environ.get("TRAIN_DONE_FILE", "").strip()
    eval_watch_poll_seconds = _get_env_int("EVAL_WATCH_POLL_SECONDS", 60)

    if n_eval_tracks <= 0:
        raise EvaluationException("N_EVAL_TRACKS must be > 0, got %d" % int(n_eval_tracks))
    if max_seconds < 0:
        raise EvaluationException("MAX_SECONDS must be >= 0, got %d" % int(max_seconds))

    eval_dir.mkdir(parents=True, exist_ok=True)

    if wandb.run is not None:
        wandb.config.update(
            {
                "n_eval_tracks": n_eval_tracks,
                "max_seconds": max_seconds,
                "device": device,
                "eval_subset": eval_subset,
                "eval_every_n_ckpts": eval_every_n_ckpts,
                "eval_chunk_frames": eval_chunk_frames,
                "eval_overlap_frames": eval_overlap_frames,
                "eval_amp": eval_amp,
            }
        )

    tracks = _list_tracks(data_root / eval_subset)[:n_eval_tracks]
    if not tracks:
        raise EvaluationException(
            "No valid %s tracks found in: %s" % (eval_subset, str(data_root / eval_subset))
        )

    # Determine checkpoint list (single-pass modes). Watch mode discovers ckpts dynamically.
    ckpts: list[Path] = []
    if not watch_mode:
        all_ckpts = sorted(ckpt_dir.glob("*.pth"), key=_ckpt_sort_key)
        if not all_ckpts:
            raise EvaluationException("No checkpoints found in: %s" % str(ckpt_dir))
        ckpts = all_ckpts[eval_every_n_ckpts - 1 :: eval_every_n_ckpts]
        if not ckpts:
            ckpts = [all_ckpts[-1]]

    logger.info(
        "Final evaluation using MUSDB subset '%s' with %d tracks.",
        eval_subset,
        int(len(tracks)),
    )
    logger.info(
        "Evaluation inference settings: chunk_frames=%d overlap_frames=%d amp=%s",
        int(eval_chunk_frames),
        int(eval_overlap_frames),
        str(bool(eval_amp)),
    )

    per_track_csv = eval_dir / "fullsong_eval_per_track.csv"
    summary_csv = eval_dir / "fullsong_eval_summary.csv"

    # Append to existing CSVs in watch mode.
    append_mode = bool(watch_mode and per_track_csv.exists() and per_track_csv.stat().st_size > 0)
    csv_mode = "a" if append_mode else "w"

    with (
        per_track_csv.open(csv_mode, newline="") as f_track,
        summary_csv.open(csv_mode, newline="") as f_sum,
    ):
        track_writer = csv.writer(f_track)
        sum_writer = csv.writer(f_sum)
        summary_rows: list[dict[str, Union[float, str]]] = []

        if not append_mode:
            track_header = ["ckpt", "track"]
            for stem in STEMS:
                track_header.append("sisdr_%s" % stem)
            track_header += ["recon_snr_db", "interstem_corr"]
            track_writer.writerow(track_header)

            sum_header = ["ckpt", "mean_sisdr"]
            for stem in STEMS:
                sum_header.append("mean_sisdr_%s" % stem)
            sum_header += [
                "mean_recon_snr_db",
                "mean_interstem_corr",
                "vocals_priority_score",
            ]
            sum_writer.writerow(sum_header)

        _flush_outputs(f_track, f_sum, do_fsync=False)

        total_tracks = int(len(tracks))
        total_ckpts = int(len(ckpts)) if not watch_mode else 0
        use_rich_progress = bool(progress_enabled and os.isatty(1) and not watch_mode)
        text_progress_enabled = bool(progress_enabled and not use_rich_progress)
        per_track_rows_written = 0

        progress_ctx = (
            create_themed_progress("{task.fields[group]:<11}", title_style=BOLD_PURPLE)
            if use_rich_progress
            else nullcontext(None)
        )
        with progress_ctx as progress:
            ckpt_task_id = None
            track_task_id = None
            ckpt_anim_stop = None
            ckpt_anim_thread = None
            track_anim_stop = None
            track_anim_thread = None
            if progress is not None:
                ckpt_task_id = progress.add_task(
                    "eval_ckpt",
                    total=total_ckpts,
                    eq="",
                    group="Checkpoints",
                )
                track_task_id = progress.add_task(
                    "eval_track",
                    total=(total_ckpts * total_tracks),
                    eq="",
                    group="Tracks",
                )
                ckpt_anim_stop, ckpt_anim_thread = start_eq_animator(progress, ckpt_task_id)
                track_anim_stop, track_anim_thread = start_eq_animator(progress, track_task_id)
            try:
                if watch_mode:
                    evaled: set[str] = set()
                    train_done_path = Path(train_done_file_str) if train_done_file_str else None
                    ckpt_counter = 0

                    def _scan_and_eval_new() -> None:
                        nonlocal per_track_rows_written, ckpt_counter
                        if not ckpt_dir.is_dir():
                            return
                        new_ckpts = []
                        all_found = sorted(ckpt_dir.glob("*.pth"), key=_ckpt_sort_key)
                        # Apply skip filter
                        filtered = all_found[eval_every_n_ckpts - 1 :: eval_every_n_ckpts]
                        if not filtered and all_found:
                            filtered = [all_found[-1]]

                        for c in filtered:
                            if str(c) in evaled:
                                continue
                            # Ensure file is finished writing (simple heuristic)
                            try:
                                if (time.time() - c.stat().st_mtime) < 10:
                                    continue
                            except OSError:
                                continue
                            new_ckpts.append(c)

                        for ckpt_path in new_ckpts:
                            ckpt_counter += 1
                            evaled.add(str(ckpt_path))
                            try:
                                row, per_track_rows_written = _eval_one_checkpoint(
                                    ckpt_path=ckpt_path,
                                    ckpt_idx=ckpt_counter,
                                    total_ckpts=-1,
                                    tracks=tracks,
                                    device=device,
                                    eval_chunk_frames=eval_chunk_frames,
                                    eval_overlap_frames=eval_overlap_frames,
                                    eval_amp=eval_amp,
                                    max_seconds=max_seconds,
                                    eval_dir=eval_dir,
                                    track_writer=track_writer,
                                    sum_writer=sum_writer,
                                    f_track=f_track,
                                    f_sum=f_sum,
                                    flush_every=flush_every,
                                    fsync_every=fsync_every,
                                    per_track_rows_written=per_track_rows_written,
                                    text_progress_enabled=text_progress_enabled,
                                    print_metrics=print_metrics,
                                    print_every_tracks=print_every_tracks,
                                    progress=progress,
                                    ckpt_task_id=ckpt_task_id,
                                    track_task_id=track_task_id,
                                )
                                summary_rows.append(row)
                            except Exception as exc:
                                logger.error(
                                    "Eval watcher: checkpoint %s failed. Error: %s",
                                    ckpt_path.name,
                                    exc,
                                    exc_info=True,
                                )

                    while True:
                        _scan_and_eval_new()
                        if train_done_path and train_done_path.exists():
                            _scan_and_eval_new()  # final scan
                            break
                        time.sleep(eval_watch_poll_seconds)

                else:
                    for ckpt_idx, ckpt_path in enumerate(ckpts, start=1):
                        row, per_track_rows_written = _eval_one_checkpoint(
                            ckpt_path=ckpt_path,
                            ckpt_idx=ckpt_idx,
                            total_ckpts=total_ckpts,
                            tracks=tracks,
                            device=device,
                            eval_chunk_frames=eval_chunk_frames,
                            eval_overlap_frames=eval_overlap_frames,
                            eval_amp=eval_amp,
                            max_seconds=max_seconds,
                            eval_dir=eval_dir,
                            track_writer=track_writer,
                            sum_writer=sum_writer,
                            f_track=f_track,
                            f_sum=f_sum,
                            flush_every=flush_every,
                            fsync_every=fsync_every,
                            per_track_rows_written=per_track_rows_written,
                            text_progress_enabled=text_progress_enabled,
                            print_metrics=print_metrics,
                            print_every_tracks=print_every_tracks,
                            progress=progress,
                            ckpt_task_id=ckpt_task_id,
                            track_task_id=track_task_id,
                        )
                        summary_rows.append(row)

            finally:
                if ckpt_anim_stop is not None and ckpt_anim_thread is not None:
                    ckpt_anim_stop.set()
                    ckpt_anim_thread.join()
                if progress is not None and ckpt_task_id is not None:
                    progress.update(ckpt_task_id, eq="▁▁▁▁▁")
                if track_anim_stop is not None and track_anim_thread is not None:
                    track_anim_stop.set()
                    track_anim_thread.join()
                if progress is not None and track_task_id is not None:
                    progress.update(track_task_id, eq="▁▁▁▁▁")

        _flush_outputs(f_track, f_sum, do_fsync=bool(int(fsync_every) > 0))

    logger.info("Wrote per-track CSV: %s", str(per_track_csv))
    logger.info("Wrote summary CSV: %s", str(summary_csv))
    _print_progress(progress_enabled, "Evaluation complete.")
    print_evaluation_summary(summary_rows)


if __name__ == "__main__":
    raise SystemExit(main())
