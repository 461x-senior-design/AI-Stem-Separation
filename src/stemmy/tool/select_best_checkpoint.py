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

Optional guard behavior:
- A baseline row can be selected from a baseline summary CSV.
- Candidate rows can then be rejected if they drop too far below baseline on:
  - any stem
  - vocals specifically
  - reconstruction SNR

Optional checkpoint resolution behavior:
- If --prefer-ckpt-dir is used, checkpoint paths are resolved from --ckpt-dir by
  epoch first, before using any checkpoint path embedded in the CSV rows.
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
    """Represents a scored checkpoint row for ranking."""

    epoch: int
    score: float
    recon_snr: Optional[float]
    corr: Optional[float]
    sisdr_values: list[float]
    ckpt_path: Optional[Path]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


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
    """Parse SI-SDR values from a comma-separated list-like string."""
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


# ---------------------------------------------------------------------------
# CSV row extraction helpers
# ---------------------------------------------------------------------------


def _extract_epoch(row: dict[str, Any]) -> Optional[int]:
    """Extract epoch from a CSV row."""
    for key in ("epoch", "Epoch", "ckpt_epoch"):
        if key in row:
            value = _parse_int(row.get(key))
            if value is not None:
                return value

    for key in ("ckpt", "ckpt_path", "checkpoint", "checkpoint_path"):
        if key in row:
            s = str(row.get(key) or "")
            match = re.search(r"epoch(\d{1,4})", s)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None

    return None


def _extract_sisdr(row: dict[str, Any]) -> list[float]:
    """Extract per-stem SI-SDR values from a CSV row."""
    for key in ("sisdr", "SI-SDR", "si_sdr"):
        if key in row:
            values = _parse_sisdr_list(row.get(key))
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
        if all(key in row for key in key_order):
            values: list[float] = []
            for key in key_order:
                value = _parse_float(row.get(key))
                if value is None:
                    return []
                values.append(value)
            return values

    return []


def _extract_recon_snr(row: dict[str, Any]) -> Optional[float]:
    """Extract reconstruction SNR from a CSV row."""
    for key in (
        "recon_snr",
        "recon_snr_db",
        "mean_recon_snr_db",
        "reconstruction_snr",
        "snr",
        "snr_db",
    ):
        if key in row:
            value = _parse_float(row.get(key))
            if value is not None:
                return value
    return None


def _extract_corr(row: dict[str, Any]) -> Optional[float]:
    """Extract correlation metric from a CSV row."""
    for key in ("corr", "correlation", "mix_corr", "corrcoef", "mean_interstem_corr"):
        if key in row:
            value = _parse_float(row.get(key))
            if value is not None:
                return value
    return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> Optional[float]:
    """Compute mean of a list."""
    if not values:
        return None
    return sum(values) / float(len(values))


def _stem_index(stem_name: str) -> int:
    """Return the canonical index for a stem name."""
    if stem_name not in STEMS_4:
        raise ValueError("Unknown stem name: %s" % stem_name)
    return list(STEMS_4).index(stem_name)


def _select_scalar_score(
    sisdr_values: list[float],
    recon_snr: Optional[float],
    corr: Optional[float],
    metric: str,
) -> Optional[float]:
    """Compute a scalar score from extracted metrics."""
    metric = metric.strip().lower()
    if metric == "":
        return None

    if metric == "mean_sisdr":
        return _mean(sisdr_values)

    if metric == "vocals_sisdr":
        if "vocals" not in STEMS_4:
            return None
        vocals_index = _stem_index("vocals")
        if vocals_index < len(sisdr_values):
            return sisdr_values[vocals_index]
        return None

    if metric == "recon_snr":
        return recon_snr

    if metric == "corr":
        return corr

    if metric == "weighted":
        mean_sisdr = _mean(sisdr_values)
        if mean_sisdr is None or recon_snr is None or corr is None:
            return None
        return mean_sisdr + 0.05 * recon_snr + 2.0 * corr

    if metric == "weighted_vocals":
        mean_sisdr = _mean(sisdr_values)
        if mean_sisdr is None:
            return None
        if "vocals" not in STEMS_4:
            return None

        vocals_index = _stem_index("vocals")
        if vocals_index >= len(sisdr_values):
            return None

        vocals = sisdr_values[vocals_index]
        recon_term = 0.0 if recon_snr is None else 0.05 * recon_snr
        corr_term = 0.0 if corr is None else 2.0 * corr
        return mean_sisdr + 0.5 * vocals + recon_term + corr_term

    if metric.startswith("sisdr_index_"):
        parts = metric.split("_")
        if not parts or parts[-1].strip() == "":
            return None
        try:
            index = int(parts[-1])
        except ValueError:
            return None
        if index < 0 or index >= len(sisdr_values):
            return None
        return sisdr_values[index]

    return None


def candidate_passes_guards(
    candidate: RowScore,
    baseline: Optional[RowScore],
    max_drop_any_stem_db: float,
    max_drop_vocals_db: float,
    min_recon_snr_db: float,
) -> bool:
    """Return True if candidate satisfies optional baseline and absolute guards."""
    if candidate.recon_snr is not None and candidate.recon_snr < min_recon_snr_db:
        return False

    if baseline is None:
        return True

    if max_drop_any_stem_db >= 0.0:
        if len(candidate.sisdr_values) != len(baseline.sisdr_values):
            return False
        for candidate_value, baseline_value in zip(
            candidate.sisdr_values, baseline.sisdr_values
        ):
            if candidate_value < (baseline_value - max_drop_any_stem_db):
                return False

    if max_drop_vocals_db >= 0.0:
        vocals_index = _stem_index("vocals")
        if vocals_index >= len(candidate.sisdr_values) or vocals_index >= len(
            baseline.sisdr_values
        ):
            return False
        if candidate.sisdr_values[vocals_index] < (
            baseline.sisdr_values[vocals_index] - max_drop_vocals_db
        ):
            return False

    return True


def select_best_row(scores: list[RowScore]) -> RowScore:
    """Return the top-ranked row from an already-scored list."""
    if not scores:
        raise RuntimeError("No scores available.")
    return sorted(scores, key=lambda row: row.score, reverse=True)[0]


# ---------------------------------------------------------------------------
# Checkpoint path resolution helpers
# ---------------------------------------------------------------------------


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


def _resolve_ckpt_path_from_row(
    row: dict[str, Any],
    ckpt_dir: Optional[Path] = None,
    prefer_ckpt_dir: bool = False,
) -> Optional[Path]:
    """Resolve checkpoint path for a scored row."""
    epoch = _extract_epoch(row)

    if prefer_ckpt_dir and ckpt_dir is not None and epoch is not None:
        ckpt_path = _find_checkpoint_for_epoch(ckpt_dir, epoch)
        if ckpt_path is not None:
            return ckpt_path

    for key in ("ckpt", "ckpt_path", "checkpoint", "checkpoint_path"):
        if key in row:
            raw = str(row.get(key) or "").strip()
            if raw == "":
                continue
            path = Path(raw).expanduser()
            try:
                path = path.resolve()
            except OSError:
                path = path.absolute()
            if path.is_file():
                return path

    if ckpt_dir is not None and epoch is not None:
        ckpt_path = _find_checkpoint_for_epoch(ckpt_dir, epoch)
        if ckpt_path is not None:
            return ckpt_path

    return None


# ---------------------------------------------------------------------------
# Loading and display helpers
# ---------------------------------------------------------------------------


def load_scores(
    summary_csv: Path,
    ckpt_dir: Optional[Path],
    metric: str,
    prefer_ckpt_dir: bool = False,
) -> list[RowScore]:
    """Load and score rows from an evaluation summary CSV."""
    if not summary_csv.is_file():
        raise FileNotFoundError("Summary CSV not found: %s" % str(summary_csv))

    rows: list[RowScore] = []
    with summary_csv.open("r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
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

            ckpt_path = _resolve_ckpt_path_from_row(
                row=row,
                ckpt_dir=ckpt_dir,
                prefer_ckpt_dir=prefer_ckpt_dir,
            )

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


def _format_optional(value: Optional[float]) -> str:
    """Format an optional float for display."""
    if value is None:
        return ""
    return f"{value:.6f}"


def _delta_list(candidate: RowScore, baseline: Optional[RowScore]) -> list[float]:
    """Compute per-stem SI-SDR deltas relative to baseline."""
    if baseline is None:
        return []
    if len(candidate.sisdr_values) != len(baseline.sisdr_values):
        return []
    return [candidate_value - baseline_value for candidate_value, baseline_value in zip(
        candidate.sisdr_values,
        baseline.sisdr_values,
    )]


def print_top(scores: list[RowScore], top_k: int, baseline: Optional[RowScore] = None) -> None:
    """Print a ranked list of top scores."""
    scores_sorted = sorted(scores, key=lambda row: row.score, reverse=True)
    print(f"Ranked checkpoints (top {top_k}):")

    if baseline is None:
        print("rank,epoch,score,recon_snr,corr,sisdr_values,ckpt_path")
    else:
        print("rank,epoch,score,recon_snr,corr,sisdr_values,delta_sisdr_values,ckpt_path")

    for i, row in enumerate(scores_sorted[:top_k], start=1):
        recon_snr_str = _format_optional(row.recon_snr)
        corr_str = _format_optional(row.corr)
        ckpt_path_str = "" if row.ckpt_path is None else str(row.ckpt_path)

        if baseline is None:
            print(
                f'{i},{row.epoch},{row.score:.6f},{recon_snr_str},{corr_str},"{row.sisdr_values}",{ckpt_path_str}'
            )
        else:
            deltas = _delta_list(row, baseline)
            print(
                f'{i},{row.epoch},{row.score:.6f},{recon_snr_str},{corr_str},'
                f'"{row.sisdr_values}","{deltas}",{ckpt_path_str}'
            )


def copy_best(best: RowScore, dest_path: Path) -> None:
    """Copy the best checkpoint to a destination path."""
    if best.ckpt_path is None:
        raise RuntimeError("Best checkpoint path is unknown (provide --ckpt-dir or CSV ckpt path).")
    if not best.ckpt_path.is_file():
        raise FileNotFoundError("Checkpoint file not found: %s" % str(best.ckpt_path))

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best.ckpt_path), str(dest_path))
    print(f"COPIED BEST: epoch={best.epoch} score={best.score:.6f}")
    print(f"FROM: {best.ckpt_path}")
    print(f"TO:   {dest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "--prefer-ckpt-dir",
        action="store_true",
        help=(
            "If set, resolve checkpoint paths from --ckpt-dir by epoch before using any "
            "checkpoint path embedded in the CSV rows."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_sisdr",
        help=(
            "Ranking metric: mean_sisdr | vocals_sisdr | recon_snr | corr | weighted "
            "| weighted_vocals | sisdr_index_N"
        ),
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
    parser.add_argument(
        "--baseline-summary-csv",
        type=str,
        default="",
        help=(
            "Optional summary CSV used to define a baseline row for guard checks. "
            "If provided, the best row from this CSV under --baseline-metric is used as baseline."
        ),
    )
    parser.add_argument(
        "--baseline-metric",
        type=str,
        default="mean_sisdr",
        help="Metric used to choose the baseline row from --baseline-summary-csv.",
    )
    parser.add_argument(
        "--max-drop-any-stem-db",
        type=float,
        default=-1.0,
        help=(
            "If >= 0, reject any candidate whose per-stem SI-SDR drops by more than this many dB "
            "below the baseline on any stem."
        ),
    )
    parser.add_argument(
        "--max-drop-vocals-db",
        type=float,
        default=-1.0,
        help=(
            "If >= 0, reject any candidate whose vocals SI-SDR drops by more than this many dB "
            "below the baseline."
        ),
    )
    parser.add_argument(
        "--min-recon-snr-db",
        type=float,
        default=float("-inf"),
        help="Reject candidates with reconstruction SNR below this value.",
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

    baseline_metric = str(args.baseline_metric).strip()
    if baseline_metric == "":
        print("ERROR: --baseline-metric cannot be empty", file=sys.stderr)
        return 2

    if args.top_k <= 0:
        print("ERROR: --top-k must be > 0", file=sys.stderr)
        return 2

    scores = load_scores(
        summary_csv=summary_csv,
        ckpt_dir=ckpt_dir,
        metric=metric,
        prefer_ckpt_dir=bool(args.prefer_ckpt_dir),
    )
    if not scores:
        print("ERROR: No valid rows were scored. Check CSV columns and --metric.", file=sys.stderr)
        return 3

    baseline: Optional[RowScore] = None
    if str(args.baseline_summary_csv).strip() != "":
        baseline_csv = Path(args.baseline_summary_csv).expanduser().resolve()
        baseline_scores = load_scores(
            summary_csv=baseline_csv,
            ckpt_dir=ckpt_dir,
            metric=baseline_metric,
            prefer_ckpt_dir=bool(args.prefer_ckpt_dir),
        )
        if not baseline_scores:
            print(
                "ERROR: No valid baseline rows were scored. "
                "Check --baseline-summary-csv and --baseline-metric.",
                file=sys.stderr,
            )
            return 3
        baseline = select_best_row(baseline_scores)

    filtered_scores = [
        row
        for row in scores
        if candidate_passes_guards(
            candidate=row,
            baseline=baseline,
            max_drop_any_stem_db=float(args.max_drop_any_stem_db),
            max_drop_vocals_db=float(args.max_drop_vocals_db),
            min_recon_snr_db=float(args.min_recon_snr_db),
        )
    ]

    if not filtered_scores:
        print(
            "ERROR: No candidates passed the guard checks. Try relaxing the thresholds.",
            file=sys.stderr,
        )
        return 4

    if baseline is not None:
        print("BASELINE:")
        print(f"epoch={baseline.epoch} score={baseline.score:.6f}")
        print(f"baseline sisdr_values: {baseline.sisdr_values}")
        if baseline.recon_snr is not None:
            print(f"baseline recon_snr: {baseline.recon_snr:.6f}")
        if baseline.corr is not None:
            print(f"baseline corr: {baseline.corr:.6f}")
        if baseline.ckpt_path is not None:
            print(f"baseline ckpt_path: {baseline.ckpt_path}")
        print("")

    print_top(filtered_scores, args.top_k, baseline=baseline)

    best = select_best_row(filtered_scores)
    print("")
    print(f"BEST: epoch={best.epoch} score={best.score:.6f}")
    print(f"BEST sisdr_values: {best.sisdr_values}")
    if best.recon_snr is not None:
        print(f"BEST recon_snr: {best.recon_snr:.6f}")
    if best.corr is not None:
        print(f"BEST corr: {best.corr:.6f}")
    if best.ckpt_path is not None:
        print(f"BEST ckpt_path: {best.ckpt_path}")

    if baseline is not None:
        deltas = _delta_list(best, baseline)
        if deltas:
            print(f"BEST delta_sisdr_values: {deltas}")

    if str(args.copy_to).strip() != "":
        dest_path = Path(args.copy_to).expanduser().resolve()
        copy_best(best, dest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
