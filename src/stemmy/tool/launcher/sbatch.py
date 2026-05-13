"""Render and submit the sbatch template for a single run."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import paths


class SbatchSubmitError(RuntimeError):
    pass


def render(
    run_id: str,
    run_dir: Path,
    config_env_path: Path,
    partition: str,
    time: str = "04:30:00",
    workdir: Path | None = None,
) -> str:
    tmpl_path = paths.sbatch_template()
    if not tmpl_path.is_file():
        raise FileNotFoundError(f"missing sbatch template: {tmpl_path}")
    tmpl = tmpl_path.read_text()
    mapping = {
        "RUN_ID": run_id,
        "RUN_DIR": str(run_dir),
        "CONFIG_ENV": str(config_env_path),
        "PARTITION": partition,
        "TIME": time,
        "WORKDIR": str(workdir or paths.project_root()),
        "TRAIN_INNER": str(paths.train_inner()),
    }
    rendered = tmpl
    for k, v in mapping.items():
        rendered = rendered.replace(f"@@{k}@@", v)
    return rendered


def submit(rendered_script: str, run_dir: Path) -> str:
    """Write the rendered script into run_dir and sbatch it. Returns job id."""
    run_dir.mkdir(parents=True, exist_ok=True)
    sbatch_file = run_dir / "submit.sbatch"
    sbatch_file.write_text(rendered_script)
    sbatch_file.chmod(0o755)
    try:
        out = subprocess.check_output(
            ["sbatch", str(sbatch_file)], text=True, stderr=subprocess.STDOUT
        )
    except FileNotFoundError as e:
        raise SbatchSubmitError("sbatch not found on PATH — run this on an HPC login node") from e
    except subprocess.CalledProcessError as e:
        raise SbatchSubmitError(f"sbatch failed:\n{e.output}") from e
    m = re.search(r"(\d+)", out)
    if not m:
        raise SbatchSubmitError(f"could not parse job id from sbatch output:\n{out}")
    return m.group(1)
