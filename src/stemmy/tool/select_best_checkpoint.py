# src/stemmy/tool/select_best_checkpoint.py
"""Utility to rank checkpoints using an evaluation summary CSV.

This script reads a summary CSV produced by the evaluation pipeline, computes a
scalar score per epoch using a chosen metric, prints a ranked list, and can
optionally copy the best checkpoint to a destination path.

Expected inputs:
- --summary-csv: CSV produced by evaluation (for example fullsong_eval_summary.csv).
- --ckpt-dir: Optional directory containing checkpoints named with epoch numbers.

Ranking:
- Each CSV row is converted into a RowScore when an epoch can be extracted and the
  chosen metric can be computed.
- Rows are sorted by score (descending). The top-k are printed.
- If --copy-to is provided, the best checkpoint is copied to that destination.
"""

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from stemmy.constants import STEMS_4


@dataclass(frozen=True)
class RowScore:
    """Represents a scored checkpoint row for ranking.

    Attributes:
        epoch: Training epoch associated with this row.
        score: Scalar score used for sorting.
        recon_snr: Reconstruction SNR value if available.
        corr: Correlation value if available.
        sisdr_values: Per-stem SI-SDR values in STEMS_4 order.
        ckpt_path: Resolved checkpoint path if known.
    """

    epoch: int
    score: float
    recon_snr: Optional[float]
    corr: Optional[float]
    sisdr_values: list[float]
    ckpt_path: Optional[Path]


def _parse_float(value: Any) -> Optional[float]:
    """Parse a float from a possibly-empty CSV cell."""
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
    """Parse an int from a possibly-empty CSV cell."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_sisdr_list(value: Any) -> list[float]:
    """Parse SI-SDR values from a comma-separated list-like string.

    Accepted formats include:
        - "1.0,2.0,3.0,4.0"
        - "[1.0, 2.0, 3.0, 4.0]"
    """
    if value is None:
        return []
    s = str(value).strip()
    if s == "":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s == "":
        return []

    out: list[float] = []
    for part in [p.strip() for p in s.split(",") if p.strip() != ""]:
        try:
            out.append(float(part))
        except ValueError:
            return []
    return out


def _extract_epoch(row: dict[str, Any]) -> Optional[int]:
    """Extract epoch from a CSV row.

    Supported sources:
    - Direct integer fields: epoch / Epoch / ckpt_epoch
    - Epoch embedded in a checkpoint path field (searching for 'epochNNN')
    """
    for k in ("epoch", "Epoch", "ckpt_epoch"):
        if k in row:
            v = _parse_int(row.get(k))
            if v is not None:
                return v

    for k in ("ckpt", "ckpt_path", "checkpoint", "checkpoint_path"):
        if k in row:
            s = str(row.get(k) or "")
            m = re.search(r"epoch(\d{1,4})", s)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None

    return None


def _extract_sisdr(row: dict[str, Any]) -> list[float]:
    """Extract per-stem SI-SDR values from a CSV row.

    Supported CSV layouts:
        - Single list-like column: "sisdr" / "SI-SDR" / "si_sdr"
        - Indexed columns: "sisdr_0".."sisdr_3"
        - Named columns in STEMS_4 order:
            "sisdr_drums".."sisdr_other"
            "mean_sisdr_drums".."mean_sisdr_other"
            "drums_sisdr".."other_sisdr"
            "mean_drums_sisdr".."mean_other_sisdr"

    Returns:
        List of SI-SDR values in STEMS_4 order, or an empty list if not found/parseable.
    """
    for k in ("sisdr", "SI-SDR", "si_sdr"):
        if k in row:
            values = _parse_sisdr_list(row.get(k))
            if values:
                return values

    indexed_values: list[float] = []
    found_indexed = False
    for i in range(len(STEMS_4)):
        key = f"sisdr_{i}"
        if key in row:
            found_indexed = True
            value = _parse_float(row.get(key))
            if value is None:
                return []
            indexed_values.append(value)
    if found_indexed and indexed_values:
        return indexed_values

    key_orders = [
        [f"sisdr_{stem}" for stem in STEMS_4],
        [f"{stem}_sisdr" for stem in STEMS_4],
        [f"mean_sisdr_{stem}" for stem in STEMS_4],
        [f"mean_{stem}_sisdr" for stem in STEMS_4],
    ]
    for key_order in key_orders:
        if all(k in row for k in key_order):
            values: list[float] = []
            for k in key_order:
                value = _parse_float(row.get(k))
                if value is None:
                    return []
                values.append(value)
            return values

    return []


def _extract_recon_snr(row: dict[str, Any]) -> Optional[float]:
    """Extract reconstruction SNR from a CSV row."""
    for k in (
        "recon_snr",
        "recon_snr_db",
        "mean_recon_snr_db",
        "reconstruction_snr",
        "snr",
        "snr_db",
    ):
        if k in row:
            v = _parse_float(row.get(k))
            if v is not None:
                return v
    return None


def _extract_corr(row: dict[str, Any]) -> Optional[float]:
    """Extract correlation metric from a CSV row."""
    for k in ("corr", "correlation", "mix_corr", "corrcoef", "mean_interstem_corr"):
        if k in row:
            v = _parse_float(row.get(k))
            if v is not None:
                return v
    return None


def _mean(values: list[float]) -> Optional[float]:
    """Compute mean of a list."""
    if not values:
        return None
    return sum(values) / float(len(values))


def _select_scalar_score(
    sisdr_values: list[float],
    recon_snr: Optional[float],
    corr: Optional[float],
    metric: str,
) -> Optional[float]:
    """Compute a scalar score from extracted metrics.

    Supported metric strings:
        - "mean_sisdr"
        - "vocals_sisdr"
        - "sisdr_index_N"
        - "recon_snr"
        - "corr"
        - "weighted"

    Notes:
        - "weighted" requires all three: mean_sisdr, recon_snr, corr.
        - "vocals_sisdr" depends on STEMS_4 containing "vocals".
    """
    metric = metric.strip().lower()
    if metric == "":
        return None

    if metric == "mean_sisdr":
        return _mean(sisdr_values)

    if metric == "vocals_sisdr":
        if "vocals" not in STEMS_4:
            return None
        vocals_index = list(STEMS_4).index("vocals")
        if vocals_index < len(sisdr_values):
            return sisdr_values[vocals_index]
        return None

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
        parts = metric.split("_")
        if not parts or parts[-1].strip() == "":
            return None
        try:
            idx = int(parts[-1])
        except ValueError:
            return None
        if idx < 0 or idx >= len(sisdr_values):
            return None
        return sisdr_values[idx]

    return None


def _find_checkpoint_for_epoch(ckpt_dir: Path, epoch: int) -> Optional[Path]:
    """Find a checkpoint file within a directory for a given epoch."""
    pattern = f"unet_phase1_epoch{epoch:03d}.pth"
    path = ckpt_dir / pattern
    if path.is_file():
        return path

    for candidate in sorted(ckpt_dir.glob(f"*epoch{epoch:03d}*.pth")):
        if candidate.is_file():
            return candidate

    for candidate in sorted(ckpt_dir.glob(f"*epoch{epoch}*.pth")):
        if candidate.is_file():
            return candidate

    return None


def _resolve_ckpt_path_from_row(row: dict[str, Any]) -> Optional[Path]:
    """Resolve checkpoint path from a summary CSV row if present."""
    for k in ("ckpt", "ckpt_path", "checkpoint", "checkpoint_path"):
        if k in row:
            raw = str(row.get(k) or "").strip()
            if raw == "":
                continue
            p = Path(raw).expanduser()
            try:
                p = p.resolve()
            except OSError:
                p = p.absolute()
            if p.is_file():
                return p
    return None


def load_scores(
    summary_csv: Path,
    ckpt_dir: Optional[Path],
    metric: str,
) -> list[RowScore]:
    """Load and score rows from an evaluation summary CSV.

    Rows that cannot produce:
      - an epoch, or
      - a scalar score for the chosen metric
    are skipped.

    Raises:
        FileNotFoundError: If summary_csv does not exist.
        RuntimeError: If the CSV has no header row.
    """
    if not summary_csv.is_file():
        raise FileNotFoundError("Summary CSV not found: %s" % str(summary_csv))

    rows: list[RowScore] = []
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row: %s" % str(summary_csv))

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

            ckpt_path = _resolve_ckpt_path_from_row(row)
            if ckpt_path is None and ckpt_dir is not None:
                ckpt_path = _find_checkpoint_for_epoch(ckpt_dir, epoch)

            rows.append(
                RowScore(
                    epoch=int(epoch),
                    score=float(score),
                    recon_snr=recon_snr,
                    corr=corr,
                    sisdr_values=sisdr_values,
                    ckpt_path=ckpt_path,
                )
            )

    return rows


def print_top(scores: list[RowScore], top_k: int) -> None:
    """Print a ranked list of top scores."""
    scores_sorted = sorted(scores, key=lambda r: r.score, reverse=True)
    print(f"Ranked checkpoints (top {top_k}):")
    print("rank,epoch,score,recon_snr,corr,sisdr_values,ckpt_path")
    for i, row in enumerate(scores_sorted[:top_k], start=1):
        recon_snr_s = "" if row.recon_snr is None else f"{row.recon_snr:.6f}"
        corr_s = "" if row.corr is None else f"{row.corr:.6f}"
        ckpt_s = "" if row.ckpt_path is None else str(row.ckpt_path)
        print(
            f"{i},{row.epoch},{row.score:.6f},"
            f"{recon_snr_s},"
            f"{corr_s},"
            f"\"{row.sisdr_values}\","
            f"{ckpt_s}"
        )


def copy_best(best: RowScore, dest_path: Path) -> None:
    """Copy the best checkpoint to a destination path.

    Raises:
        RuntimeError: If best.ckpt_path is not available.
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if best.ckpt_path is None:
        raise RuntimeError("Best checkpoint path is unknown (provide --ckpt-dir or CSV ckpt path).")
    if not best.ckpt_path.is_file():
        raise FileNotFoundError("Checkpoint file not found: %s" % str(best.ckpt_path))

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best.ckpt_path), str(dest_path))
    print(f"COPIED BEST: epoch={best.epoch} score={best.score:.6f}")
    print(f"FROM: {best.ckpt_path}")
    print(f"TO:   {dest_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Select and optionally copy the best checkpoint based on an eval summary CSV."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        required=True,
        help="Path to fullsong_eval_summary.csv produced by the evaluation script.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="",
        help=(
            "Directory containing checkpoints, for example .../checkpoints. "
            "If provided, the script can auto-resolve the checkpoint path for each epoch."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_sisdr",
        help="Ranking metric: mean_sisdr | vocals_sisdr | recon_snr | corr | weighted | sisdr_index_N",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many ranked rows to print.",
    )
    parser.add_argument(
        "--copy-to",
        type=str,
        default="",
        help=(
            "If set, copy the best checkpoint to this path, "
            "for example runs/best_ckpt/unet_phase1_best.pth."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    summary_csv = Path(args.summary_csv).expanduser().resolve()

    ckpt_dir: Optional[Path] = None
    if str(args.ckpt_dir).strip() != "":
        ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
        if not ckpt_dir.is_dir():
            print("ERROR: --ckpt-dir is not a directory: %s" % str(ckpt_dir), file=sys.stderr)
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

