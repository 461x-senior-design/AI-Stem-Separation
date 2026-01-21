import argparse
import math
import os
from typing import Tuple

import numpy as np


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        import soundfile as sf  # type: ignore
        x, sr = sf.read(path, always_2d=True)
        return x.astype(np.float64), int(sr)
    except Exception:
        from scipy.io import wavfile  # type: ignore

        sr, x = wavfile.read(path)
        if x.ndim == 1:
            x = x[:, None]
        x = x.astype(np.float64)
        if np.issubdtype(x.dtype, np.integer):
            x = x / float(np.iinfo(x.dtype).max)
        return x, int(sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Audio must be shaped (N, C).")
    return np.mean(x, axis=1)


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    n = int(min(len(est), len(ref)))
    if n <= 0:
        raise ValueError("Empty audio after alignment.")
    est = est[:n] - float(np.mean(est[:n]))
    ref = ref[:n] - float(np.mean(ref[:n]))

    ref_energy = float(np.dot(ref, ref)) + eps
    alpha = float(np.dot(est, ref)) / ref_energy
    s_target = alpha * ref
    e_noise = est - s_target

    num = float(np.dot(s_target, s_target)) + eps
    den = float(np.dot(e_noise, e_noise)) + eps
    return 10.0 * math.log10(num / den)


def snr_db(sig: np.ndarray, noise: np.ndarray, eps: float = 1e-9) -> float:
    n = int(min(len(sig), len(noise)))
    if n <= 0:
        raise ValueError("Empty audio after alignment.")
    sig = sig[:n]
    noise = noise[:n]
    p_sig = float(np.dot(sig, sig)) + eps
    p_noise = float(np.dot(noise, noise)) + eps
    return 10.0 * math.log10(p_sig / p_noise)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross SI-SDR matrix: predicted stems vs GT stems (MUSDB).")
    parser.add_argument("--track-dir", required=True, help="MUSDB track dir containing drums.wav/bass.wav/vocals.wav/other.wav")
    parser.add_argument("--out-dir", required=True, help="Inference output dir containing drums.wav/bass.wav/vocals.wav/other.wav")
    args = parser.parse_args()

    track_dir = os.path.abspath(os.path.expanduser(args.track_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    if not os.path.isdir(track_dir):
        raise SystemExit(f"--track-dir is not a directory: {track_dir}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"--out-dir is not a directory: {out_dir}")

    stems = ["drums", "bass", "vocals", "other"]

    gt = {}
    pr = {}
    sr_ref = None

    for s in stems:
        gt_path = os.path.join(track_dir, f"{s}.wav")
        pr_path = os.path.join(out_dir, f"{s}.wav")
        if not os.path.isfile(gt_path):
            raise SystemExit(f"Missing GT stem: {gt_path}")
        if not os.path.isfile(pr_path):
            raise SystemExit(f"Missing predicted stem: {pr_path}")

        gt_x, gt_sr = read_wav(gt_path)
        pr_x, pr_sr = read_wav(pr_path)

        if gt_sr != pr_sr:
            raise SystemExit(f"Sample-rate mismatch for {s}: gt={gt_sr} pred={pr_sr}")

        if sr_ref is None:
            sr_ref = gt_sr
        elif gt_sr != sr_ref:
            raise SystemExit(f"Inconsistent GT sample-rates: expected {sr_ref}, got {gt_sr}")

        gt[s] = to_mono(gt_x)
        pr[s] = to_mono(pr_x)

    # Reconstruction check: sum(pred stems) vs mixture.wav if present
    mix_path = os.path.join(track_dir, "mixture.wav")
    if os.path.isfile(mix_path):
        mix_x, mix_sr = read_wav(mix_path)
        if mix_sr != sr_ref:
            raise SystemExit(f"mixture.wav SR mismatch: mix={mix_sr} stems={sr_ref}")
        mix = to_mono(mix_x)
        pred_sum = pr["drums"] + pr["bass"] + pr["vocals"] + pr["other"]
        recon_noise = mix[: min(len(mix), len(pred_sum))] - pred_sum[: min(len(mix), len(pred_sum))]
        recon = snr_db(mix, recon_noise)
        print(f"Reconstruction SNR (mono, mixture vs sum(pred stems)): {recon:.3f} dB\n")

    print("Cross SI-SDR (pred -> GT):")
    print("GT column order:", stems)

    for ps in stems:
        best_gs = None
        best_score = None
        row_scores = []
        for gs in stems:
            score = si_sdr(pr[ps], gt[gs])
            row_scores.append(score)
            if best_score is None or score > best_score:
                best_score = score
                best_gs = gs
        row = "  ".join([f"{v:8.3f}" for v in row_scores])
        print(f"{ps:6s}: {row}   | best match: {best_gs} ({best_score:.3f} dB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
