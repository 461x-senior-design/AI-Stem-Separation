"""
Centralized Wandb configuration for AI-Stem-Separation project.

This module provides a single source of truth for wandb settings
used across training and evaluation scripts.

Projects:
    - ai-stem: Development/experimentation project
    - ai-stem-production: Production/final models project

Set WANDB_ENV=production to use ai-stem-production, otherwise uses ai-stem (dev).

Usage with decorators:
    @wandb_run(job_type="training", name="my-run")
    def train(...):
        wandb.log({"loss": loss})
"""

import functools
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

import wandb


class WandbEnvironment(Enum):
    """Wandb project environments."""

    DEV = "dev"
    PRODUCTION = "production"


# Default wandb settings
WANDB_ENTITY = "ai-stem"

# Project names by environment
WANDB_PROJECTS = {
    WandbEnvironment.DEV: "ai-stem",
    WandbEnvironment.PRODUCTION: "ai-stem-production",
}


def get_wandb_environment() -> WandbEnvironment:
    """
    Get the current wandb environment from WANDB_ENV.

    Returns:
        WandbEnvironment.PRODUCTION if WANDB_ENV=production, else DEV.
    """
    env = os.environ.get("WANDB_ENV", "dev").lower()
    if env == "production":
        return WandbEnvironment.PRODUCTION
    return WandbEnvironment.DEV


def get_wandb_project(env: Optional[WandbEnvironment] = None) -> str:
    """
    Get the wandb project name for the given environment.

    Args:
        env: Environment to get project for. If None, reads from WANDB_ENV.

    Returns:
        Project name string.
    """
    if env is None:
        env = get_wandb_environment()
    return WANDB_PROJECTS[env]


# Default project (based on environment)
WANDB_PROJECT = get_wandb_project()


@dataclass
class WandbConfig:
    """Configuration for wandb initialization."""

    entity: str = WANDB_ENTITY
    project: str = WANDB_PROJECT
    name: Optional[str] = None
    job_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    resume: Optional[str] = None


def init_wandb(
    cfg: WandbConfig,
    enabled: bool = True,
) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Initialize wandb with the given configuration.

    Args:
        cfg: WandbConfig object with settings.
        enabled: If False, skip wandb initialization.

    Returns:
        wandb Run object if successful, None otherwise.
    """
    if not enabled:
        return None

    # Append date to run name
    date_str = datetime.now().strftime("%Y%m%d")
    if cfg.name:
        run_name = f"{cfg.name}_{date_str}"
    else:
        run_name = date_str

    try:
        run = wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=run_name,
            job_type=cfg.job_type,
            config=cfg.config,
            resume=cfg.resume,
        )
        print(f"Wandb initialized: entity={cfg.entity}, project={cfg.project}, run={run.name}")
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Continuing without wandb logging...")
        return None


def finish_wandb(run: Optional[wandb.sdk.wandb_run.Run]) -> None:
    """Finish wandb run if it exists."""
    if run is not None:
        run.finish()


def get_wandb_config_from_env(
    default_project: Optional[str] = None,
    default_entity: str = WANDB_ENTITY,
) -> tuple[str, str, str, bool]:
    """
    Get wandb configuration from environment variables.

    Environment variables:
        WANDB_ENV: "dev" or "production" (default: dev)
        WANDB_PROJECT: Override project name (optional)
        WANDB_ENTITY: Entity/team name - ai-stem
        WANDB_RUN_NAME: Run name (default: empty)
        NO_WANDB: Set to "1" to disable wandb

    Returns:
        Tuple of (project, entity, run_name, enabled)
    """
    # Use environment-based project if not explicitly set
    if default_project is None:
        default_project = get_wandb_project()

    project = os.environ.get("WANDB_PROJECT", default_project)
    entity = os.environ.get("WANDB_ENTITY", default_entity)
    run_name = os.environ.get("WANDB_RUN_NAME", "")
    no_wandb = os.environ.get("NO_WANDB", "0") == "1"

    return project, entity, run_name, not no_wandb


def wandb_run(
    job_type: str = "training",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to wrap a function with wandb run initialization and cleanup.

    Usage:
        @wandb_run(job_type="training", name="baseline")
        def train(args):
            wandb.log({"loss": loss})

        @wandb_run(job_type="evaluation", name="song_eval")
        def evaluate():
            wandb.log({"sisdr": sisdr})

    Args:
        job_type: Type of job (training, evaluation, etc.)
        name: Run name (date will be appended automatically)
        config: Config dict to log to wandb

    Returns:
        Decorated function that auto-initializes and finishes wandb.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get config from environment
            project, entity, env_run_name, enabled = get_wandb_config_from_env()

            # Use provided name, env name, or default based on job_type
            run_name = name or env_run_name or job_type

            # Initialize wandb
            wandb_cfg = WandbConfig(
                entity=entity,
                project=project,
                name=run_name,
                job_type=job_type,
                config=config,
            )
            run = init_wandb(wandb_cfg, enabled=enabled)

            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)
                return result
            finally:
                # Always finish wandb run
                finish_wandb(run)

        return wrapper

    return decorator
