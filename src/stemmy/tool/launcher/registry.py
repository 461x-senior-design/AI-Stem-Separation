"""The run.json registry: create/read/update/compare entries under runs/."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import gitx, paths


@dataclass
class RunEntry:
    run_id: str
    name: str
    created_at: str
    status: str = "pending"
    git: dict[str, Any] = field(default_factory=dict)
    config: dict[str, str] = field(default_factory=dict)
    config_sources: dict[str, str] = field(default_factory=dict)
    config_name: str | None = None
    sbatch: dict[str, Any] = field(default_factory=dict)
    wandb: dict[str, Any] = field(default_factory=dict)
    resume_of: str | None = None
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return paths.runs_root() / self.run_id

    def write(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "run.json"
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        (self.dir / "status.txt").write_text(self.status + "\n")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_id(name: str, sha: str | None = None) -> str:
    sha = sha or gitx.current_sha()
    return f"{name}_{_stamp()}_{gitx.short_sha(sha)}"


def new_entry(
    name: str,
    config_values: dict[str, str],
    config_sources: dict[str, str],
    config_name: str | None,
    resume_of: str | None = None,
) -> RunEntry:
    sha = gitx.current_sha()
    entry = RunEntry(
        run_id=make_run_id(name, sha=sha),
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git={
            "branch": gitx.current_branch(),
            "sha": sha,
            "dirty": gitx.is_dirty(),
        },
        config=dict(config_values),
        config_sources=dict(config_sources),
        config_name=config_name,
        resume_of=resume_of,
    )
    return entry


def load(run_id: str) -> RunEntry:
    p = paths.runs_root() / run_id / "run.json"
    if not p.is_file():
        raise FileNotFoundError(f"no run.json at {p}")
    data = json.loads(p.read_text())
    return RunEntry(**data)


def find(pattern: str) -> RunEntry:
    """Match by exact run_id or by name prefix (most-recent wins)."""
    root = paths.runs_root()
    exact = root / pattern / "run.json"
    if exact.is_file():
        return load(pattern)
    candidates = sorted(
        (p for p in root.glob(f"{pattern}*/run.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no runs matching '{pattern}' under {root}")
    return load(candidates[0].parent.name)


def list_all() -> list[RunEntry]:
    root = paths.runs_root()
    if not root.is_dir():
        return []
    out: list[RunEntry] = []
    for p in sorted(root.glob("*/run.json")):
        try:
            out.append(load(p.parent.name))
        except Exception:
            continue
    return out


def update_status(run_id: str, status: str) -> None:
    entry = load(run_id)
    entry.status = status
    entry.write()


def update_metrics_from_eval_csv(entry: RunEntry) -> None:
    """Refresh metrics_snapshot from eval_summary.csv if present."""
    csv_path = entry.dir / "eval" / "fullsong_eval_summary.csv"
    if not csv_path.is_file():
        return
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    best = max(rows, key=lambda r: _float(r.get("mean_sisdr")))
    last = rows[-1]
    entry.metrics_snapshot = {
        "best_sisdr": _float(best.get("mean_sisdr")),
        "best_epoch": _int(best.get("epoch")),
        "final_sisdr": _float(last.get("mean_sisdr")),
        "final_epoch": _int(last.get("epoch")),
        "n_checkpoints_evaluated": len(rows),
    }
    entry.write()


def _float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("-inf")


def _int(v) -> int:
    try:
        return int(v)
    except Exception:
        return -1
