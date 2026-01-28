import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RowScore:
    epoch: int
    score: float
    recon_snr: Optional[float]
    corr: Optional[float]
    sisdr_values: List[float]
    ckpt_path: Optional[Path]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_sisdr_list(value: Any) -> List[float]:
    """
    Accepts:
      - "[-5.77,-17.49,-22.85,-8.01]"
      - "[-5.77, -17.49, -22.85, -8.01]"
      - "-5.77,-17.49,-22.85,-8.01"
    Returns [] if not parseable.
    """
    if value is None:
        return []
    s = str(value).strip()
    if s == "":
        return []
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            return []
    return out


def _extract_epoch(row: Dict[str, Any]) -> Optional[int]:
    # Most common
    for k in ("epoch", "Epoch", "ckpt_epoch"):
        if k in row:
            v = _parse_int(row.get(k))
            if v is not None:
                return v

    # Fallback: parse from a ckpt filename column if present
    for k in ("ckpt", "ckpt_path", "checkpoint", "checkpoint_path"):
        if k in row:
            s = str(row.get(k) or "")
            m = re.search(r"epoch(\d{1,4})", s)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass

    return None


def _extract_sisdr(row: Dict[str, Any]) -> List[float]:
    # 1) single column "sisdr"
    for k in ("sisdr", "SI-SDR", "si_sdr"):
        if k in row:
            vals = _parse_sisdr_list(row.get(k))
            if vals:
                return vals

    # 2) per-stem columns: sisdr_0..sisdr_3
    vals2: List[float] = []
    found_any = False
    for i in range(0, 8):  # allow up to 8 stems; you use 4
        key = f"sisdr_{i}"
        if key in row:
            found_any = True
            v = _parse_float(row.get(key))
            if v is None:
                return []
            vals2.append(v)
    if found_any and vals2:
        return vals2

    # 3) named columns (varies by implementation)
    # Try common stem names
    cand_keys = [
        "sisdr_drums",
        "sisdr_bass",
        "sisdr_vocals",
        "sisdr_other",
        "drums_sisdr",
        "bass_sisdr",
        "vocals_sisdr",
        "other_sisdr",
    ]
    vals3: List[float] = []
    any_named = False
    for k in cand_keys:
        if k in row:
            any_named = True
            v = _parse_float(row.get(k))
            if v is None:
                return []
            vals3.append(v)
    if any_named and vals3:
        return vals3

    return []


def _extract_recon_snr(row: Dict[str, Any]) -> Optional[float]:
    for k in ("recon_snr", "recon_snr_db", "reconstruction_snr", "snr", "snr_db"):
        if k in row:
            v = _parse_float(row.get(k))
            if v is not None:
                return v
    return None


def _extract_corr(row: Dict[str, Any]) -> Optional[float]:
    for k in ("corr", "correlation", "mix_corr", "corrcoef"):
        if k in row:
            v = _parse_float(row.get(k))
            if v is not None:
                return v
    return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _select_scalar_score(
    sisdr_values: List[float],
    recon_snr: Optional[float],
    corr: Optional[float],
    metric: str,
) -> Optional[float]:
    """
    metric options:
      - mean_sisdr
      - vocals_sisdr (index 2 by your ordering in logs: [drums, bass, other, vocals] is unknown,
        but your printed list order appears consistent across runs. If you want a different index,
        use --sisdr-index.)
      - sisdr_index_N (e.g., sisdr_index_0)
      - recon_snr
      - corr
      - weighted (mean_sisdr + 0.05*recon_snr + 2.0*corr)
    """
    metric = metric.strip().lower()

    if metric == "mean_sisdr":
        return _mean(sisdr_values)

    if metric == "recon_snr":
        return recon_snr

    if metric == "corr":
        return corr

    if metric == "weighted":
        m = _mean(sisdr_values)
        if m is None or recon_snr is None or corr is None:
            return None
        return m + 0.05 * recon_snr + 2.0 * corr

    if metric.startswith("sisdr_index_"):
        try:
            idx = int(metric.split("_")[-1])
        except ValueError:
            return None
        if idx < 0 or idx >= len(sisdr_values):
            return None
        return sisdr_values[idx]

    if metric == "vocals_sisdr":
        # Default assumption: 4-stem and vocals is index 2 or 3 depending on your ordering.
        # Your project’s stem ordering should be made explicit; for now this is not used unless requested.
        return None

    return None


def _find_checkpoint_for_epoch(ckpt_dir: Path, epoch: int) -> Optional[Path]:
    # Matches your naming: unet_phase1_epoch210.pth
    pattern = f"unet_phase1_epoch{epoch:03d}.pth"
    p = ckpt_dir / pattern
    if p.is_file():
        return p

    # Fallback: search
    candidates = sorted(ckpt_dir.glob(f"*epoch{epoch:03d}*.pth"))
    for c in candidates:
        if c.is_file():
            return c
    candidates = sorted(ckpt_dir.glob(f"*epoch{epoch}*.pth"))
    for c in candidates:
        if c.is_file():
            return c

    return None


def load_scores(
    summary_csv: Path,
    ckpt_dir: Optional[Path],
    metric: str,
) -> List[RowScore]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    rows: List[RowScore] = []
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header row: {summary_csv}")

        for row in reader:
            epoch = _extract_epoch(row)
            if epoch is None:
                continue

            sisdr_values = _extract_sisdr(row)
            recon_snr = _extract_recon_snr(row)
            corr = _extract_corr(row)

            score = _select_scalar_score(sisdr_values, recon_snr, corr, metric)
            if score is None:
                continue

            ckpt_path: Optional[Path] = None
            if ckpt_dir is not None:
                ckpt_path = _find_checkpoint_for_epoch(ckpt_dir, epoch)

            rows.append(
                RowScore(
                    epoch=epoch,
                    score=float(score),
                    recon_snr=recon_snr,
                    corr=corr,
                    sisdr_values=sisdr_values,
                    ckpt_path=ckpt_path,
                )
            )

    return rows


def print_top(scores: List[RowScore], top_k: int) -> None:
    scores_sorted = sorted(scores, key=lambda r: r.score, reverse=True)
    print(f"Ranked checkpoints (top {top_k}):")
    print("rank,epoch,score,recon_snr,corr,sisdr_values,ckpt_path")
    for i, r in enumerate(scores_sorted[:top_k], start=1):
        print(
            f"{i},{r.epoch},{r.score:.6f},"
            f"{'' if r.recon_snr is None else f'{r.recon_snr:.6f}'},"
            f"{'' if r.corr is None else f'{r.corr:.6f}'},"
            f'"{r.sisdr_values}",'
            f"{'' if r.ckpt_path is None else str(r.ckpt_path)}"
        )


def copy_best(best: RowScore, dest_path: Path) -> None:
    if best.ckpt_path is None:
        raise RuntimeError("Best checkpoint path is unknown (provide --ckpt-dir).")
    if not best.ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {best.ckpt_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best.ckpt_path), str(dest_path))
    print(f"COPIED BEST: epoch={best.epoch} score={best.score:.6f}")
    print(f"FROM: {best.ckpt_path}")
    print(f"TO:   {dest_path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select and optionally copy the best checkpoint based on an eval summary CSV."
    )
    p.add_argument(
        "--summary-csv",
        type=str,
        required=True,
        help="Path to fullsong_eval_summary.csv produced by your eval script.",
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default="",
        help="Directory containing checkpoints (e.g., .../checkpoints). If provided, can auto-copy winner.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="mean_sisdr",
        help="Ranking metric: mean_sisdr | recon_snr | corr | weighted | sisdr_index_N",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many ranked rows to print.",
    )
    p.add_argument(
        "--copy-to",
        type=str,
        default="",
        help="If set, copy the best checkpoint to this path (e.g., runs/best_ckpt/unet_phase1_best.pth).",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    summary_csv = Path(args.summary_csv).expanduser().resolve()
    ckpt_dir: Optional[Path] = None
    if str(args.ckpt_dir).strip() != "":
        ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
        if not ckpt_dir.is_dir():
            print(f"ERROR: --ckpt-dir is not a directory: {ckpt_dir}", file=sys.stderr)
            return 2

    metric = str(args.metric).strip()
    if metric == "":
        print("ERROR: --metric cannot be empty", file=sys.stderr)
        return 2

    if args.top_k <= 0:
        print("ERROR: --top-k must be > 0", file=sys.stderr)
        return 2

    scores = load_scores(summary_csv, ckpt_dir, metric)
    if not scores:
        print("ERROR: No valid rows were scored. Check CSV columns and --metric.", file=sys.stderr)
        return 3

    print_top(scores, args.top_k)

    best = sorted(scores, key=lambda r: r.score, reverse=True)[0]
    print("")
    print(f"BEST: epoch={best.epoch} score={best.score:.6f}")
    print(f"BEST sisdr_values: {best.sisdr_values}")
    if best.recon_snr is not None:
        print(f"BEST recon_snr: {best.recon_snr:.6f}")
    if best.corr is not None:
        print(f"BEST corr: {best.corr:.6f}")
    if best.ckpt_path is not None:
        print(f"BEST ckpt_path: {best.ckpt_path}")

    if str(args.copy_to).strip() != "":
        dest_path = Path(args.copy_to).expanduser().resolve()
        copy_best(best, dest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
