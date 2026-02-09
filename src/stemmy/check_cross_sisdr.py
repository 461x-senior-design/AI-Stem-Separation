# src/stemmy/check_cross_sisdr.py
"""Cross SI-SDR matrix: predicted stems vs ground-truth stems (MUSDB-style folders).

Given:
--track-dir: MUSDB track directory containing {drums,bass,vocals,other}.wav
(and optionally mixture.wav)
--out-dir: inference output directory containing predicted stems

Prints:
- Optional reconstruction SNR (mixture vs sum(pred stems), mono)
- Cross SI-SDR matrix (pred -> GT) and best match per predicted stem
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from stemmy.constants import STEMS_4


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a wav file as float64 with shape (N, C) and return (audio, sample_rate).

    Args:
        path: Path to the wav file.

    Returns:
        Tuple of:
        - audio: np.ndarray float64 shaped (N, C)
        - sample_rate: int
    """
    wav_path = Path(path)
    if not wav_path.is_file():
        raise FileNotFoundError(str(wav_path))

    try:
        import soundfile as sf  # type: ignore

        x, sr = sf.read(str(wav_path), always_2d=True)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
        else:
            x = x.astype(np.float64)
        return x, int(sr)
    except Exception:
        from scipy.io import wavfile  # type: ignore

        sr, x = wavfile.read(str(wav_path))
        if x.ndim == 1:
            x = x[:, None]
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
        else:
            x = x.astype(np.float64)
        return x, int(sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to mono.

    Args:
        x: Audio shaped (N, C).

    Returns:
        Mono audio shaped (N,).
    """
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Audio must have shape (N, C).")
    return np.mean(x, axis=1)


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    """Compute scale-invariant SDR (SI-SDR) in dB between estimated and reference signals.

    Args:
        est: Estimated mono audio shaped (N,).
        ref: Reference mono audio shaped (N,).
        eps: Small constant to avoid division by zero.

    Returns:
        SI-SDR value in dB.
    """
    n = int(min(len(est), len(ref)))
    if n <= 0:
        raise ValueError("Empty audio after alignment.")

    est_zm = est[:n] - float(np.mean(est[:n]))
    ref_zm = ref[:n] - float(np.mean(ref[:n]))

    ref_energy = float(np.dot(ref_zm, ref_zm)) + float(eps)
    alpha = float(np.dot(est_zm, ref_zm)) / ref_energy

    s_target = alpha * ref_zm
    e_noise = est_zm - s_target

    num = float(np.dot(s_target, s_target)) + float(eps)
    den = float(np.dot(e_noise, e_noise)) + float(eps)
    return 10.0 * math.log10(num / den)


def snr_db(sig: np.ndarray, noise: np.ndarray, eps: float = 1e-9) -> float:
    """Compute SNR (signal-to-noise ratio) in dB.

    Args:
        sig: Signal audio shaped (N,).
        noise: Noise audio shaped (N,).
        eps: Small constant to avoid division by zero.

    Returns:
        SNR value in dB.
    """
    n = int(min(len(sig), len(noise)))
    if n <= 0:
        raise ValueError("Empty audio after alignment.")

    sig_n = sig[:n]
    noise_n = noise[:n]
    p_sig = float(np.dot(sig_n, sig_n)) + float(eps)
    p_noise = float(np.dot(noise_n, noise_n)) + float(eps)
    return 10.0 * math.log10(p_sig / p_noise)


def resolve_pred_path(out_dir: Path, stem: str, stem_name: Optional[str]) -> Path:
    """Resolve predicted stem path from an inference output directory.

    Supports:
    - <out_dir>/<stem>.wav
    - <out_dir>/<stem_name>_<stem>.wav
    - If neither exists, and exactly one match for "*_<stem>.wav" exists, use it.
    - If multiple matches exist, require --stem-name to disambiguate.

    Args:
        out_dir: Output directory containing predicted stems
        stem: Stem name (e.g., "drums")
        stem_name: Optional base name used in exported files

    Returns:
        Path to the predicted stem wav file
    """
    direct = out_dir / f"{stem}.wav"
    if direct.is_file():
        return direct

    if stem_name is not None:
        named = out_dir / f"{stem_name}_{stem}.wav"
        if named.is_file():
            return named

    matches = sorted(out_dir.glob(f"*_{stem}.wav"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = "\n".join(str(p) for p in matches[:25])
        more = "" if len(matches) <= 25 else f"\n... and {len(matches) - 25} more"
        raise SystemExit(
            f"Multiple candidate predicted files for stem '{stem}' under {out_dir}:\n"
            f"{choices}{more}\n"
            "Provide --stem-name to disambiguate."
        )

    expected_named = f"<stem_name>_{stem}.wav"
    raise SystemExit(
        f"Missing predicted stem for '{stem}' under {out_dir}. "
        f"Expected either '{stem}.wav' or '{expected_named}'."
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross SI-SDR matrix: predicted stems vs GT stems (MUSDB)."
    )
    parser.add_argument(
        "--track-dir",
        required=True,
        help="MUSDB track dir containing drums.wav/bass.wav/vocals.wav/other.wav",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Inference output dir containing predicted stems",
    )
    parser.add_argument(
        "--stem-name",
        default="",
        help="Base name used in exported predictions (for '<stem_name>_<stem>.wav').",
    )
    args = parser.parse_args()

    track_dir = Path(os.path.abspath(os.path.expanduser(args.track_dir)))
    out_dir = Path(os.path.abspath(os.path.expanduser(args.out_dir)))

    if not track_dir.is_dir():
        raise SystemExit(f"--track-dir is not a directory: {track_dir}")
    if not out_dir.is_dir():
        raise SystemExit(f"--out-dir is not a directory: {out_dir}")

    stems = list(STEMS_4)

    stem_name_arg = args.stem_name.strip()
    stem_name: Optional[str] = stem_name_arg if stem_name_arg else None

    gt: dict[str, np.ndarray] = {}
    pr: dict[str, np.ndarray] = {}
    sr_ref: Optional[int] = None

    for stem in stems:
        gt_path = track_dir / f"{stem}.wav"
        pr_path = resolve_pred_path(out_dir, stem, stem_name)

        if not gt_path.is_file():
            raise SystemExit(f"Missing GT stem: {gt_path}")
        if not pr_path.is_file():
            raise SystemExit(f"Missing predicted stem: {pr_path}")

        gt_x, gt_sr = read_wav(str(gt_path))
        pr_x, pr_sr = read_wav(str(pr_path))

        if gt_sr != pr_sr:
            raise SystemExit(
                f"Sample-rate mismatch for {stem}: gt={gt_sr} pred={pr_sr} "
                f"(pred file: {pr_path})"
            )

        if sr_ref is None:
            sr_ref = gt_sr
        elif gt_sr != sr_ref:
            raise SystemExit(f"Inconsistent GT sample-rates: expected {sr_ref}, got {gt_sr}")

        gt[stem] = to_mono(gt_x)
        pr[stem] = to_mono(pr_x)

    mix_path = track_dir / "mixture.wav"
    if mix_path.is_file():
        mix_x, mix_sr = read_wav(str(mix_path))
        if sr_ref is not None and mix_sr != sr_ref:
            raise SystemExit(f"mixture.wav SR mismatch: mix={mix_sr} stems={sr_ref}")
        mix = to_mono(mix_x)
        pred_sum = pr["drums"] + pr["bass"] + pr["vocals"] + pr["other"]
        n = min(len(mix), len(pred_sum))
        recon_noise = mix[:n] - pred_sum[:n]
        recon = snr_db(mix[:n], recon_noise)
        print(f"Reconstruction SNR (mono, mixture vs sum(pred stems)): {recon:.3f} dB\n")

    print("Cross SI-SDR (pred -> GT):")
    print("GT column order:", stems)

    for pred_stem in stems:
        best_gt: Optional[str] = None
        best_score: Optional[float] = None
        row_scores: list[float] = []

        for gt_stem in stems:
            score = si_sdr(pr[pred_stem], gt[gt_stem])
            row_scores.append(score)
            if best_score is None or score > best_score:
                best_score = score
                best_gt = gt_stem

        row = " ".join([f"{v:8.3f}" for v in row_scores])
        print(f"{pred_stem:6s}: {row} | best match: {best_gt} ({best_score:.3f} dB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

