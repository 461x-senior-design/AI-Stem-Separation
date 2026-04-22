"""Path resolution: project root, scripts dir, configs, registry."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Locate the repo root containing pyproject.toml.

    Honors STEMMY_REPO first. Otherwise walks up from CWD, then from this
    file, looking for a directory that contains pyproject.toml.
    """
    override = os.environ.get("STEMMY_REPO")
    if override:
        return Path(override).expanduser().resolve()

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate

    raise RuntimeError(
        "could not locate repo root (no pyproject.toml found above CWD or the "
        "launcher package). Set STEMMY_REPO to pin it."
    )


def scripts_dir() -> Path:
    """scripts/v2/ directory — where all v2 scripts and configs live."""
    return project_root() / "scripts" / "v2"


def defaults_env() -> Path:
    return scripts_dir() / "defaults.env"


def configs_dir() -> Path:
    return scripts_dir() / "configs"


def shared_configs_dir() -> Path:
    return configs_dir() / "shared"


def user_name() -> str:
    """Namespace key for per-user configs. STEMMY_USER > USER > ONID."""
    for var in ("STEMMY_USER", "USER", "ONID"):
        v = os.environ.get(var)
        if v:
            return v
    return "unknown"


def user_configs_dir() -> Path:
    return configs_dir() / user_name()


def runs_root() -> Path:
    """Local run registry (run.json + eval summaries). Heavy artifacts live
    on HPC scratch via RUNS_BASE; see registry.py."""
    return project_root() / "runs"


def sbatch_template() -> Path:
    return scripts_dir() / "stemmy.sbatch.tmpl"


def train_inner() -> Path:
    return scripts_dir() / "_train_inner.sh"


def matrices_dir() -> Path:
    return scripts_dir() / "matrices"
