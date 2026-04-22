"""Env-file layer resolution.

Layers (earliest first, later wins):
  1. scripts/defaults.env
  2. scripts/configs/<user>/defaults.env       (if present)
  3. scripts/configs/<user>/<name>.env  OR     scripts/configs/shared/<name>.env
  4. CLI KEY=VALUE overrides

Sourcing is delegated to bash so `${VAR}` expansion, quoting, and
comments all work the way the existing env files expect.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import paths


@dataclass
class ResolvedConfig:
    """Merged env with per-key provenance."""

    values: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    layers: list[tuple[str, Path]] = field(default_factory=list)

    def to_env_lines(self) -> list[str]:
        return [f'{k}="{v}"' for k, v in sorted(self.values.items())]

    def to_env_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.to_env_lines()) + "\n")


def _find_named_config(name: str) -> Path | None:
    """User dir wins over shared on collision."""
    candidates = [
        paths.user_configs_dir() / f"{name}.env",
        paths.shared_configs_dir() / f"{name}.env",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def config_is_shared(path: Path) -> bool:
    try:
        return paths.shared_configs_dir() in path.parents
    except Exception:
        return False


def list_configs() -> dict[str, dict]:
    """Return {name: {shared: Path|None, user: Path|None}}.
    `user` shadows `shared` during resolution."""
    out: dict[str, dict] = {}
    for p in sorted(paths.shared_configs_dir().glob("*.env")):
        out.setdefault(p.stem, {"shared": None, "user": None})["shared"] = p
    ud = paths.user_configs_dir()
    if ud.is_dir():
        for p in sorted(ud.glob("*.env")):
            if p.name == "defaults.env":
                continue
            out.setdefault(p.stem, {"shared": None, "user": None})["user"] = p
    return out


def resolve_layers(
    config_name_or_path: str | None,
    cli_overrides: dict[str, str] | None = None,
) -> ResolvedConfig:
    """Resolve the full env. `config_name_or_path` may be a named config
    (searched in user dir then shared/) or a raw path to an .env file."""
    layers: list[tuple[str, Path]] = []

    defaults = paths.defaults_env()
    if defaults.is_file():
        layers.append(("defaults", defaults))

    user_defaults = paths.user_configs_dir() / "defaults.env"
    if user_defaults.is_file():
        layers.append((f"user:{paths.user_name()}/defaults", user_defaults))

    if config_name_or_path:
        p = Path(config_name_or_path)
        if p.suffix == ".env" and p.exists():
            layers.append((f"path:{p}", p.resolve()))
        else:
            found = _find_named_config(config_name_or_path)
            if found is None:
                raise FileNotFoundError(
                    f"config '{config_name_or_path}' not found in "
                    f"{paths.user_configs_dir()} or {paths.shared_configs_dir()}"
                )
            label = "user" if not config_is_shared(found) else "shared"
            layers.append((f"{label}:{found.name}", found))

    values, sources = _source_and_diff(layers)

    if cli_overrides:
        for k, v in cli_overrides.items():
            values[k] = v
            sources[k] = "cli"

    return ResolvedConfig(values=values, sources=sources, layers=layers)


def _source_and_diff(
    layers: list[tuple[str, Path]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Source each layer in order and track which layer set each final value."""
    if not layers:
        return {}, {}
    proj = paths.project_root()
    parts = ["set -a"]
    parts.append("printf '__STEMMY_LAYER__:baseline\\n'")
    parts.append("env")
    for label, path in layers:
        safe_label = label.replace("\n", " ")
        # Prefer a path that bash can resolve from project_root. On Windows
        # with WSL bash, absolute C:/ paths break `source`; a relative path
        # lets bash inherit the translated cwd.
        try:
            rel = os.path.relpath(str(path), str(proj))
            src_path = rel.replace("\\", "/")
        except ValueError:
            src_path = str(path)
        parts.append(f"printf '__STEMMY_LAYER__:{safe_label}\\n'")
        parts.append(f"source {shlex.quote(src_path)}")
        parts.append("env")
    script = "\n".join(parts)
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=str(proj),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to source env layers (rc={proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    snapshots: list[tuple[str, dict[str, str]]] = []
    current_label: str | None = None
    current_env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("__STEMMY_LAYER__:"):
            if current_label is not None:
                snapshots.append((current_label, current_env))
            current_label = line.split(":", 1)[1]
            current_env = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            if k.isidentifier() or all(c.isalnum() or c == "_" for c in k):
                current_env[k] = v
    if current_label is not None:
        snapshots.append((current_label, current_env))

    ignore = {
        "PATH",
        "PWD",
        "SHLVL",
        "_",
        "OLDPWD",
        "SHELL",
        "HOME",
        "IFS",
        "PS1",
        "PS2",
    }
    baseline = snapshots[0][1] if snapshots else {}
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for label, env in snapshots[1:]:
        for k, v in env.items():
            if k in ignore:
                continue
            prev = values.get(k, baseline.get(k))
            if prev != v:
                values[k] = v
                sources[k] = label
    return values, sources
