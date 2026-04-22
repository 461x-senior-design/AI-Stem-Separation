"""Thin wrappers around git for SHA / branch / dirty-tree lookups."""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import paths


def _git(*args: str, cwd: Path | None = None) -> str:
    cwd = cwd or paths.project_root()
    out = subprocess.check_output(
        ["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT
    )
    return out.strip()


def current_sha() -> str:
    return _git("rev-parse", "HEAD")


def short_sha(sha: str | None = None) -> str:
    return _git("rev-parse", "--short", sha or "HEAD")


def current_branch() -> str:
    try:
        return _git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return "HEAD"


def is_dirty() -> bool:
    out = _git("status", "--porcelain")
    return bool(out.strip())
