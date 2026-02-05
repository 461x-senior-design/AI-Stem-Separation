import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

import wandb
from src.inference import config_from_checkpoint, load_pth_model
from src.training.stft import freq_minmax_normalize
from src.wandb_config import wandb_run

STEMS = ["drums", "bass", "vocals", "other"]


def list_tracks(split_dir: Path) -> List[Path]:
    tracks: List[Path] = []
    for p in sorted(split_dir.iterdir()):
        if p.is_dir():
            mix = p / "mixture.wav"
            ok = mix.exists()
            for s in STEMS:
                ok = ok and (p / f"{s}.wav").exists()
            if ok:
                tracks.append(p)
    return tracks


def zero_mean(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    est = est.astype(np.float64)
    ref = ref.astype(np.float64)
    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch est={est.shape} ref={ref.shape}")

    est = zero_mean(est)
    ref = zero_mean(ref)

    sdrs: List[float] = []
    for c in range(ref.shape[1]):
        r = ref[:, c]
        e = est[:, c]
        rr = float(np.dot(r, r) + eps)
        a = float(np.dot(e, r) / rr)
        s_target = a * r
        e_noise = e - s_target
        num = float(np.dot(s_target, s_target) + eps)
        den = float(np.dot(e_noise, e_noise) + eps)
        sdrs.append(10.0 * math.log10(num / den))
    return float(np.mean(sdrs))


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def recon_snr_db(mix: np.ndarray, stems_sum: np.ndarray, eps: float = 1e-12) -> float:
    mix = mix.astype(np.float64)
    stems_sum = stems_sum.astype(np.float64)
    err = mix - stems_sum
    mix_pow = float(np.mean(mix * mix) + eps)
    err_pow = float(np.mean(err * err) + eps)
    return float(10.0 * math.log10(mix_pow / err_pow))


def corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = (np.sqrt(np.mean(a * a)) * np.sqrt(np.mean(b * b))) + eps
    return float(np.mean(a * b) / den)


def mean_interstem_corr(pred_stems: Dict[str, np.ndarray]) -> float:
    vals: List[float] = []
    for i in range(len(STEMS)):
        for j in range(i + 1, len(STEMS)):
            vals.append(corr(pred_stems[STEMS[i]], pred_stems[STEMS[j]]))
    return float(np.mean(vals)) if vals else 0.0


def epoch_from_name(p: Path) -> int:
    s = p.stem
    if "epoch" in s:
        tail = s.split("epoch")[-1]
        try:
            return int(tail)
        except Exception:
            return -1
    return -1


def read_slice(path: Path, max_samples: int) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if max_samples > 0 and x.shape[0] > max_samples:
        x = x[:max_samples, :]
    return x, sr


def separate_by_masking(mix: np.ndarray, cfg, model, device: str) -> Dict[str, np.ndarray]:
    """
    Mask in STFT domain then iSTFT each stem.

    IMPORTANT: force center=True for stft/istft to avoid CUDA istft NOLA/COLA failures
    with Hann windows when cfg.center is False.
    """
    x = torch.from_numpy(mix).to(dtype=torch.float32, device=device).t().contiguous()  # [2,N]
    x_mono = x.mean(dim=0)  # [N]
    n_orig = int(x_mono.shape[-1])

    window = torch.hann_window(cfg.win_length, periodic=True, dtype=torch.float32, device=device)

    mix_stft = torch.stft(
        x_mono,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        return_complex=True,
    )  # [F,T] complex

    mix_mag = torch.abs(mix_stft)
    mix_norm, _, _ = freq_minmax_normalize(mix_mag, eps=cfg.eps)
    inp = mix_norm.unsqueeze(0).unsqueeze(0)  # [1,1,F,T]

    with torch.no_grad():
        out = model(inp)[0]  # [S,F,T]

    eps = 1e-8
    m = torch.clamp(out, 0.0, 1.0)
    m = m / (m.sum(dim=0, keepdim=True) + eps)  # sum-to-1 per bin

    stems_stft = m * mix_stft.unsqueeze(0)  # [S,F,T] complex

    pred: Dict[str, np.ndarray] = {}
    for i, name in enumerate(STEMS):
        y = torch.istft(
            stems_stft[i],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=True,
            length=n_orig,
        )  # [N]
        y_stereo = y.unsqueeze(1).repeat(1, 2)  # [N,2]
        pred[name] = y_stereo.detach().cpu().numpy()

    return pred


@wandb_run(job_type="evaluation", name="song_eval")
def evaluate() -> None:
    """Main evaluation function wrapped with wandb decorator."""
    data_root = Path(os.environ["DATA"])
    ckpt_dir = Path(os.environ["CKPT_DIR"])
    eval_dir = Path(os.environ["EVAL_DIR"])
    device = os.environ.get("DEVICE", "cuda")
    n_eval_tracks = int(os.environ.get("N_EVAL_TRACKS", "30"))
    max_seconds = int(os.environ.get("MAX_SECONDS", "30"))

    eval_dir.mkdir(parents=True, exist_ok=True)

    # Log config to wandb
    if wandb.run is not None:
        wandb.config.update(
            {
                "n_eval_tracks": n_eval_tracks,
                "max_seconds": max_seconds,
                "device": device,
            }
        )

    tracks = list_tracks(data_root / "test")[:n_eval_tracks]
    if not tracks:
        raise RuntimeError(f"No valid test tracks found in: {data_root / 'test'}")

    ckpts = sorted(ckpt_dir.glob("*.pth"), key=lambda p: epoch_from_name(p))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in: {ckpt_dir}")

    per_track_csv = eval_dir / "fullsong_eval_per_track.csv"
    summary_csv = eval_dir / "fullsong_eval_summary.csv"

    with per_track_csv.open("w", newline="") as f_pt:
        w_pt = csv.writer(f_pt)
        w_pt.writerow(
            [
                "epoch",
                "ckpt",
                "track",
                "sisdr_drums",
                "sisdr_bass",
                "sisdr_vocals",
                "sisdr_other",
                "recon_snr_db",
                "mean_interstem_corr",
                "pred_rms_drums",
                "pred_rms_bass",
                "pred_rms_vocals",
                "pred_rms_other",
                "seconds_used",
            ]
        )

        summary_rows: List[dict] = []

        for ckpt_path in ckpts:
            epoch = epoch_from_name(ckpt_path)
            model, ckpt = load_pth_model(str(ckpt_path), device=device, stems=4)
            cfg = config_from_checkpoint(ckpt)
            model.eval()

            max_samples = int(cfg.sample_rate * max_seconds) if max_seconds > 0 else 0
            seconds_used = float(max_seconds) if max_seconds > 0 else 0.0

            sisdr_all = {s: [] for s in STEMS}
            recon_snrs: List[float] = []
            corrs: List[float] = []

            for td in tracks:
                mix, sr = read_slice(td / "mixture.wav", max_samples)
                if sr != cfg.sample_rate:
                    raise RuntimeError(
                        f"SR mismatch in {td.name}: file={sr}, expected={cfg.sample_rate}"
                    )

                gt: Dict[str, np.ndarray] = {}
                for s in STEMS:
                    gt[s], sr2 = read_slice(td / f"{s}.wav", max_samples)
                    if sr2 != sr:
                        raise RuntimeError(f"SR mismatch in {td.name}/{s}.wav")

                pred_np = separate_by_masking(mix, cfg, model, device=device)

                for s in STEMS:
                    sisdr_all[s].append(si_sdr(pred_np[s], gt[s]))

                stems_sum = (
                    pred_np["drums"] + pred_np["bass"] + pred_np["vocals"] + pred_np["other"]
                )
                recon_snrs.append(recon_snr_db(mix, stems_sum))
                corrs.append(mean_interstem_corr(pred_np))

                w_pt.writerow(
                    [
                        epoch,
                        str(ckpt_path),
                        td.name,
                        float(sisdr_all["drums"][-1]),
                        float(sisdr_all["bass"][-1]),
                        float(sisdr_all["vocals"][-1]),
                        float(sisdr_all["other"][-1]),
                        float(recon_snrs[-1]),
                        float(corrs[-1]),
                        rms(pred_np["drums"]),
                        rms(pred_np["bass"]),
                        rms(pred_np["vocals"]),
                        rms(pred_np["other"]),
                        seconds_used if seconds_used > 0 else float(mix.shape[0]) / float(sr),
                    ]
                )

                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            row = {
                "epoch": epoch,
                "ckpt": str(ckpt_path),
                "mean_sisdr_drums": float(np.mean(sisdr_all["drums"]))
                if sisdr_all["drums"]
                else float("nan"),
                "mean_sisdr_bass": float(np.mean(sisdr_all["bass"]))
                if sisdr_all["bass"]
                else float("nan"),
                "mean_sisdr_vocals": float(np.mean(sisdr_all["vocals"]))
                if sisdr_all["vocals"]
                else float("nan"),
                "mean_sisdr_other": float(np.mean(sisdr_all["other"]))
                if sisdr_all["other"]
                else float("nan"),
                "mean_recon_snr_db": float(np.mean(recon_snrs)) if recon_snrs else float("nan"),
                "mean_interstem_corr": float(np.mean(corrs)) if corrs else float("nan"),
                "seconds_used": seconds_used,
            }
            summary_rows.append(row)

            print(
                f"epoch={epoch:03d} "
                f"sisdr=[{row['mean_sisdr_drums']:.2f},"
                f"{row['mean_sisdr_bass']:.2f},"
                f"{row['mean_sisdr_vocals']:.2f},"
                f"{row['mean_sisdr_other']:.2f}] "
                f"recon_snr={row['mean_recon_snr_db']:.2f} "
                f"corr={row['mean_interstem_corr']:.6f} "
                f"seconds_used={row['seconds_used']:.0f}"
            )

            # Log to wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/mean_sisdr_drums": row["mean_sisdr_drums"],
                        "eval/mean_sisdr_bass": row["mean_sisdr_bass"],
                        "eval/mean_sisdr_vocals": row["mean_sisdr_vocals"],
                        "eval/mean_sisdr_other": row["mean_sisdr_other"],
                        "eval/mean_recon_snr_db": row["mean_recon_snr_db"],
                        "eval/mean_interstem_corr": row["mean_interstem_corr"],
                        "time/seconds_used": row["seconds_used"],
                        "checkpoint/epoch": epoch,
                    },
                    step=epoch,
                )

            del model, ckpt
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    with summary_csv.open("w", newline="") as f_sum:
        w = csv.writer(f_sum)
        w.writerow(
            [
                "epoch",
                "ckpt",
                "mean_sisdr_drums",
                "mean_sisdr_bass",
                "mean_sisdr_vocals",
                "mean_sisdr_other",
                "mean_recon_snr_db",
                "mean_interstem_corr",
                "seconds_used",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["epoch"],
                    r["ckpt"],
                    r["mean_sisdr_drums"],
                    r["mean_sisdr_bass"],
                    r["mean_sisdr_vocals"],
                    r["mean_sisdr_other"],
                    r["mean_recon_snr_db"],
                    r["mean_interstem_corr"],
                    r["seconds_used"],
                ]
            )

    print("WROTE:", str(per_track_csv))
    print("WROTE:", str(summary_csv))


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
