"""
Minimal brightai.logging shim for local development without CodeArtifact.
This provides a compatible interface with the real brightai-logging package.
"""

import logging
import sys
from enum import Enum


class LogLevels(Enum):
    """Log levels enum matching brightai.logging.LogLevels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def setup_logging(
    force_json: bool = False,
    root_logger_level: LogLevels = LogLevels.INFO,
    logger_levels: dict | None = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        force_json: If True, use JSON formatting (not implemented in shim)
        root_logger_level: Root logger level
        logger_levels: Dictionary mapping logger names to log levels
    """
    # Configure root logger
    logging.basicConfig(
        level=root_logger_level.value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )

    # Configure specific loggers
    if logger_levels:
        for logger_name, level in logger_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level.value)

    logging.info("Logging configured (using local shim - brightai-logging not available)")
