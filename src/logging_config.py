"""Centralized logging configuration for the entire application.

All logging setup should be done through this module.
Call setup_logging() once at application startup.
"""

import logging
import os
import sys


def setup_logging(level: str = None) -> None:
    """Configure logging for the entire application.

    This should be called once at application startup (e.g., in main/CLI entry point).

    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR).
               If None, reads from LOG_LEVEL environment variable (default: INFO)
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Suppress verbose logs from third-party libraries
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
