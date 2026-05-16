"""Click command definitions for `stemmy`."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import click

from . import config as cfg
from . import paths, registry, sbatch

# ---------- helpers ----------


def _parse_kv(pairs: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise click.BadParameter(f"override '{p}' must be KEY=VALUE")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def _echo_err(msg: str) -> None:
    click.secho(msg, fg="red", err=True)


# ---------- root group ----------


@click.group()
@click.version_option("0.1.0", prog_name="stemmy")
def cli() -> None:
    """Unified driver for stemmy training runs."""


# ---------- run ----------


@cli.command()
@click.argument("config", required=False)
@click.option("--name", required=True, help="Short label for the run.")
@click.option("--partition", default="dgxh", show_default=True)
@click.option("--time", default="04:30:00", show_default=True, help="HPC walltime.")
@click.option("--dry-run", is_flag=True, help="Resolve and print; do not submit.")
@click.option(
    "--override",
    "-O",
    "overrides",
    multiple=True,
    help="KEY=VALUE override, repeatable.",
)
def run(
    config: str | None,
    name: str,
    partition: str,
    time: str,
    dry_run: bool,
    overrides: tuple[str, ...],
) -> None:
    """Submit one training run from a named config or .env path."""
    cli_overrides = _parse_kv(overrides)
    resolved = cfg.resolve_layers(config, cli_overrides)

    entry = registry.new_entry(
        name=name,
        config_values=resolved.values,
        config_sources=resolved.sources,
        config_name=config,
    )
    entry.dir.mkdir(parents=True, exist_ok=True)
    rendered_env = entry.dir / "config.env"
    resolved.to_env_file(rendered_env)

    script = sbatch.render(
        run_id=entry.run_id,
        run_dir=entry.dir,
        config_env_path=rendered_env,
        partition=partition,
        time=time,
    )
    if dry_run:
        click.echo("--- resolved config ---")
        for k in sorted(resolved.values):
            click.echo(f"{k}={resolved.values[k]}  # {resolved.sources.get(k, '?')}")
        click.echo("\n--- sbatch script ---")
        click.echo(script)
        entry.status = "dry-run"
        entry.write()
        return

    job_id = sbatch.submit(script, entry.dir)
    entry.sbatch = {"job_id": job_id, "partition": partition}
    entry.status = "submitted"
    entry.write()
    click.secho(f"submitted {entry.run_id} as SLURM job {job_id}", fg="green")


# ---------- ls / show / compare ----------


@cli.command(name="ls")
@click.option("--name", "name_pat", default=None)
@click.option("--status", "status_pat", default=None)
@click.option(
    "--sort",
    "sort_key",
    type=click.Choice(["created_at", "best_sisdr", "name"]),
    default="created_at",
)
def ls_cmd(name_pat: str | None, status_pat: str | None, sort_key: str) -> None:
    """List runs in the local registry."""
    runs = registry.list_all()
    for r in runs:
        registry.update_metrics_from_eval_csv(r)
    if name_pat:
        runs = [r for r in runs if name_pat in r.name]
    if status_pat:
        runs = [r for r in runs if r.status == status_pat]
    if sort_key == "best_sisdr":
        runs.sort(
            key=lambda r: r.metrics_snapshot.get("best_sisdr", float("-inf")),
            reverse=True,
        )
    elif sort_key == "name":
        runs.sort(key=lambda r: r.name)
    else:
        runs.sort(key=lambda r: r.created_at, reverse=True)

    if not runs:
        click.echo("(no runs)")
        return

    header = (
        f"{'run_id':44} {'name':18} {'status':12} {'best_sisdr':>11} {'epoch':>6} {'branch':20}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for r in runs:
        click.echo(
            f"{r.run_id:44} {r.name[:18]:18} {r.status[:12]:12} "
            f"{_fmt(r.metrics_snapshot.get('best_sisdr')):>11} "
            f"{_fmt(r.metrics_snapshot.get('best_epoch'), int):>6} "
            f"{r.git.get('branch', '')[:20]:20}"
        )


def _fmt(v, cast=float) -> str:
    if v is None:
        return "-"
    try:
        if cast is int:
            return f"{int(v)}"
        f = float(v)
        if f == float("-inf"):
            return "-"
        return f"{f:.2f}"
    except Exception:
        return str(v)


@cli.command()
@click.argument("run_id")
def show(run_id: str) -> None:
    """Dump a run's config, metrics, and console log tail."""
    entry = registry.find(run_id)
    registry.update_metrics_from_eval_csv(entry)
    click.secho(f"{entry.run_id}", fg="cyan", bold=True)
    click.echo(f"  name:       {entry.name}")
    click.echo(f"  status:     {entry.status}")
    click.echo(f"  created:    {entry.created_at}")
    click.echo(
        f"  git:        {entry.git.get('branch')} @ {entry.git.get('sha', '')[:12]}"
        + (" (dirty)" if entry.git.get("dirty") else "")
    )
    if entry.resume_of:
        click.echo(f"  resume_of:  {entry.resume_of}")
    if entry.sbatch:
        click.echo(
            f"  slurm:      job={entry.sbatch.get('job_id')} "
            f"partition={entry.sbatch.get('partition')}"
        )
    if entry.wandb:
        click.echo(f"  wandb:      {entry.wandb.get('url', entry.wandb)}")
    if entry.metrics_snapshot:
        click.echo("  metrics:")
        for k, v in entry.metrics_snapshot.items():
            click.echo(f"    {k}: {v}")
    click.echo("  config:")
    for k in sorted(entry.config):
        click.echo(f"    {k}={entry.config[k]}  # {entry.config_sources.get(k, '?')}")
    log = entry.dir / "logs" / "console.log"
    if log.is_file():
        click.echo("\n--- console.log (tail) ---")
        tail = log.read_text(errors="replace").splitlines()[-20:]
        click.echo("\n".join(tail))


@cli.command()
@click.argument("run_ids", nargs=-1, required=True)
def compare(run_ids: tuple[str, ...]) -> None:
    """Side-by-side config diff + final metrics for 2+ runs."""
    entries = [registry.find(r) for r in run_ids]
    for e in entries:
        registry.update_metrics_from_eval_csv(e)
    all_keys = sorted({k for e in entries for k in e.config})
    diff_keys = [k for k in all_keys if len({e.config.get(k) for e in entries}) > 1]

    col_w = max(24, max((len(e.run_id) for e in entries), default=24))
    click.secho("-- differing config keys --", bold=True)
    header = "key".ljust(30) + "".join(e.run_id[:col_w].ljust(col_w + 2) for e in entries)
    click.echo(header)
    for k in diff_keys:
        row = k.ljust(30) + "".join(
            (e.config.get(k, "-") or "-")[:col_w].ljust(col_w + 2) for e in entries
        )
        click.echo(row)

    click.secho("\n-- final metrics --", bold=True)
    metric_keys = sorted({k for e in entries for k in e.metrics_snapshot})
    click.echo("metric".ljust(30) + "".join(e.run_id[:col_w].ljust(col_w + 2) for e in entries))
    for k in metric_keys:
        row = k.ljust(30) + "".join(
            _fmt(e.metrics_snapshot.get(k)).ljust(col_w + 2) for e in entries
        )
        click.echo(row)


# ---------- config group ----------


@cli.group()
def config() -> None:
    """Manage named config files under scripts/configs/."""


@config.command("new")
@click.argument("name")
@click.option("--from", "from_", default=None, help="Seed from another named config.")
@click.option("--shared", is_flag=True, help="Create in scripts/configs/shared/.")
def config_new(name: str, from_: str | None, shared: bool) -> None:
    target_dir = paths.shared_configs_dir() if shared else paths.user_configs_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.env"
    if target.exists():
        raise click.ClickException(f"{target} already exists")

    if from_:
        src = cfg._find_named_config(from_)
        if not src:
            raise click.ClickException(f"base config '{from_}' not found")
        shutil.copyfile(src, target)
    else:
        target.write_text(f"# {name}.env — overrides on top of scripts/defaults.env\n")

    click.secho(f"created {target}", fg="green")
    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.call([editor, str(target)])
    except FileNotFoundError:
        click.echo(f"(edit manually — $EDITOR={editor} not found)")


@config.command("ls")
def config_ls() -> None:
    configs = cfg.list_configs()
    if not configs:
        click.echo("(no configs)")
        return
    click.echo(f"user:   {paths.user_name()}")
    click.echo(f"{'name':24} {'shared':8} {'user':8} shadow")
    click.echo("-" * 60)
    for name, info in sorted(configs.items()):
        s_mark = "yes" if info["shared"] else "-"
        u_mark = "yes" if info["user"] else "-"
        shadow = "user wins" if info["user"] and info["shared"] else ""
        click.echo(f"{name:24} {s_mark:8} {u_mark:8} {shadow}")


@config.command("show")
@click.argument("name")
def config_show(name: str) -> None:
    resolved = cfg.resolve_layers(name)
    click.echo(f"# config: {name}")
    click.echo(f"# user: {paths.user_name()}")
    click.echo("# layers (in order applied):")
    for label, path in resolved.layers:
        click.echo(f"#   {label}: {path}")
    click.echo()
    for k in sorted(resolved.values):
        click.echo(f'{k}="{resolved.values[k]}"  # {resolved.sources.get(k, "?")}')


@config.command("edit")
@click.argument("name")
@click.option("--shared", is_flag=True)
def config_edit(name: str, shared: bool) -> None:
    target = (
        paths.shared_configs_dir() / f"{name}.env"
        if shared
        else paths.user_configs_dir() / f"{name}.env"
    )
    if not target.exists():
        found = cfg._find_named_config(name)
        if not found:
            raise click.ClickException(f"config '{name}' not found")
        if cfg.config_is_shared(found) and not shared:
            raise click.ClickException(f"'{name}' is a shared config — pass --shared to edit it")
        target = found
    editor = os.environ.get("EDITOR", "vi")
    subprocess.call([editor, str(target)])


# ---------- matrix ----------


@cli.command()
@click.argument("matrix_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True)
def matrix(matrix_yaml: str, dry_run: bool) -> None:
    """Submit one run per cell of a YAML matrix (cartesian sweep)."""
    spec = _load_matrix_spec(Path(matrix_yaml))
    name_prefix = spec.get("name", Path(matrix_yaml).stem)
    base_configs = _base_configs(spec.get("base"))
    sweep: dict[str, list] = spec.get("sweep", {})
    partition = spec.get("partition", "dgxh")
    time = spec.get("time", "04:30:00")

    combos = _cartesian(sweep)
    total_runs = len(base_configs) * len(combos)
    click.echo(f"{total_runs} runs will be submitted")
    submitted: list[tuple[str, str]] = []
    for i, (base_config, combo) in enumerate(
        (base_config, combo) for base_config in base_configs for combo in combos
    ):
        overrides = {k: str(v) for k, v in combo.items()}
        resolved = cfg.resolve_layers(base_config, overrides)
        base_label = _label_part(base_config) if base_config else "default"
        label = f"{name_prefix}-{base_label}-{i:02d}"
        entry = registry.new_entry(
            name=label,
            config_values=resolved.values,
            config_sources=resolved.sources,
            config_name=base_config,
        )
        entry.dir.mkdir(parents=True, exist_ok=True)
        rendered_env = entry.dir / "config.env"
        resolved.to_env_file(rendered_env)
        script = sbatch.render(
            run_id=entry.run_id,
            run_dir=entry.dir,
            config_env_path=rendered_env,
            partition=partition,
            time=time,
        )
        if dry_run:
            click.echo(
                f"[dry-run] would submit {entry.run_id} "
                f"(base={base_config or '-'}, overrides={combo})"
            )
            entry.status = "dry-run"
            entry.write()
            continue
        job_id = sbatch.submit(script, entry.dir)
        entry.sbatch = {"job_id": job_id, "partition": partition}
        entry.status = "submitted"
        entry.write()
        submitted.append((entry.run_id, job_id))
        click.secho(
            f"  {entry.run_id}  job={job_id}  base={base_config or '-'} overrides={combo}",
            fg="green",
        )

    if submitted:
        click.echo(f"\nsubmitted {len(submitted)} jobs")


def _cartesian(sweep: dict[str, list]) -> list[dict]:
    keys = list(sweep.keys())
    if not keys:
        return [{}]
    out: list[dict] = [{}]
    for k in keys:
        new: list[dict] = []
        for existing in out:
            for v in sweep[k]:
                cp = dict(existing)
                cp[k] = v
                new.append(cp)
        out = new
    return out


def _base_configs(base: object) -> list[str | None]:
    if base is None or base == "":
        return [None]
    if isinstance(base, list):
        return [str(v) for v in base]
    return [str(base)]


def _label_part(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in value)


def _load_matrix_spec(path: Path) -> dict:
    """Parse the small matrix YAML subset used by scripts/v2/matrices/.

    The HPC login environment must be able to expand and submit matrices
    without relying on optional Python packages. Supported shape:

      name: run-name
      base: recon-p10
      base: [recon-p10, kl-p10]
      partition: dgxh
      sweep:
        KEY: [value1, value2]
    """
    spec: dict[str, object] = {}
    sweep: dict[str, list[str]] = {}
    in_sweep = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not raw_line.startswith((" ", "\t")):
            in_sweep = False
            key, value = _parse_matrix_kv(line)
            if key == "sweep":
                if value:
                    raise click.ClickException(f"{path}: sweep must be a mapping")
                in_sweep = True
                spec["sweep"] = sweep
            elif key == "base" and value.startswith("["):
                spec[key] = _parse_matrix_list(value)
            else:
                spec[key] = value
            continue

        if not in_sweep:
            raise click.ClickException(f"{path}: unexpected indented line: {raw_line}")

        key, value = _parse_matrix_kv(line.strip())
        sweep[key] = _parse_matrix_list(value)

    if "sweep" not in spec:
        spec["sweep"] = sweep
    return spec


def _parse_matrix_kv(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise click.ClickException(f"invalid matrix line: {line}")
    key, value = line.split(":", 1)
    key = key.strip()
    if not key:
        raise click.ClickException(f"invalid matrix line: {line}")
    return key, value.strip()


def _parse_matrix_list(value: str) -> list[str]:
    if not (value.startswith("[") and value.endswith("]")):
        raise click.ClickException(f"matrix sweep values must be inline lists: {value}")
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip("\"'") for item in inner.split(",")]
