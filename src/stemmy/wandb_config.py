"""Centralized Weights & Biases configuration for the Stemmy project.

Provides a reusable decorator (@wandb_run) that wraps any entry-point function
with automatic wandb lifecycle management: init before the function, finish in a
finally block so cleanup is guaranteed even if the function raises.

Projects:
    dev:        ai-stem          (default)
    production: ai-stem-production

Environment variables
---------------------
WANDB_ENV       : "dev" or "production" — selects the project bucket (default: "dev")
WANDB_PROJECT   : override the resolved project name directly
WANDB_ENTITY    : override the entity/team name (default: "ai-stem")
WANDB_RUN_NAME  : override the auto-generated run name
NO_WANDB        : set to "1" to skip wandb entirely (training/eval still run normally)

Usage
-----
    from stemmy.wandb_config import wandb_run
    import wandb

    @wandb_run(job_type="training", name="baseline")
    def train(args):
        ...
        if wandb.run is not None:
            wandb.log({"loss": loss}, step=step)
"""

import functools
import os
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import wandb
from stemmy.logging_config import get_logger

logger = get_logger(__name__)


class WandbEnvironment(Enum):
    """Wandb project environments."""

    DEV = "dev"
    PRODUCTION = "production"


WANDB_ENTITY = "ai-stem"

WANDB_PROJECTS: Dict[WandbEnvironment, str] = {
    WandbEnvironment.DEV: "ai-stem",
    WandbEnvironment.PRODUCTION: "ai-stem-production",
}


def make_wandb_run_name(base_name: Optional[str] = None) -> str:
    """Return a W&B display name with date and random 5-digit suffix."""
    date_str = datetime.now().strftime("%Y%m%d")
    suffix = random.randint(0, 99999)
    if base_name:
        return f"{base_name}_{date_str}_{suffix:05d}"
    return f"{date_str}_{suffix:05d}"


def get_wandb_environment() -> WandbEnvironment:
    """Read WANDB_ENV and return the matching environment enum."""
    env = os.environ.get("WANDB_ENV", "dev").lower()
    if env == "production":
        return WandbEnvironment.PRODUCTION
    return WandbEnvironment.DEV


def get_wandb_project(env: Optional[WandbEnvironment] = None) -> str:
    """Return the wandb project name for the given (or current) environment."""
    if env is None:
        env = get_wandb_environment()
    return WANDB_PROJECTS[env]


@dataclass
class WandbConfig:
    """Settings passed to wandb.init()."""

    entity: str = WANDB_ENTITY
    project: str = WANDB_PROJECTS[WandbEnvironment.DEV]
    name: Optional[str] = None
    job_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    resume: Optional[str] = None


def init_wandb(cfg: WandbConfig, enabled: bool = True) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialize a wandb run.

    Appends today's date and a random 5-digit suffix to prevent collisions.
    If wandb.init() fails, logs a warning and returns None so callers can
    continue without experiment tracking.

    Args:
        cfg:     WandbConfig with project/entity/name settings.
        enabled: When False, returns None immediately (NO_WANDB mode).

    Returns:
        Active wandb Run, or None if disabled or initialization failed.
    """
    if not enabled:
        return None

    run_name = make_wandb_run_name(cfg.name)

    try:
        run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=run_name,
            job_type=cfg.job_type,
            config=cfg.config,
            resume=cfg.resume,
            settings=wandb.Settings(_disable_stats=True),
        )
        logger.info(
            "wandb initialized: entity=%s project=%s run=%s",
            cfg.entity,
            cfg.project,
            getattr(run, "name", "unknown"),
        )
        return run
    except Exception as exc:
        logger.warning("Failed to initialize wandb (%s) — continuing without logging.", exc)
        return None


def finish_wandb(run: Optional["wandb.sdk.wandb_run.Run"]) -> None:
    """Finish the wandb run if one is active."""
    if run is not None:
        run.finish()


def _config_from_env(
    default_entity: str = WANDB_ENTITY,
) -> Tuple[str, str, str, bool]:
    """Read wandb settings from environment variables.

    Returns:
        (project, entity, run_name, enabled)
    """
    project = os.environ.get("WANDB_PROJECT", get_wandb_project())
    entity = os.environ.get("WANDB_ENTITY", default_entity)
    run_name = os.environ.get("WANDB_RUN_NAME", "")
    enabled = os.environ.get("NO_WANDB", "0") != "1"
    return project, entity, run_name, enabled


def wandb_run(
    job_type: str = "training",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator that wraps a function with wandb lifecycle management.

    On each call the decorator:
      1. Reads env vars (so the same decorated function works in any environment).
      2. Calls init_wandb() to start a run.
      3. Executes the wrapped function.
      4. Calls finish_wandb() in a finally block — cleanup is guaranteed.

    The wrapped function can call wandb.log() freely; guard with
    ``if wandb.run is not None:`` so it degrades gracefully when NO_WANDB=1.

    Args:
        job_type: W&B job type tag ("training", "evaluation", …).
        name:     Base run name; today's date is appended automatically.
        config:   Static config dict logged at run creation time.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            project, entity, env_name, enabled = _config_from_env()
            run_name = env_name or name or job_type

            cfg = WandbConfig(
                entity=entity,
                project=project,
                name=run_name,
                job_type=job_type,
                config=config,
            )
            run = init_wandb(cfg, enabled=enabled)

            try:
                return func(*args, **kwargs)
            finally:
                finish_wandb(run)

        return wrapper

    return decorator
