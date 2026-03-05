from fastapi import FastAPI

from brightai.ai_copilot.core.application import initialize_app
from brightai.ai_copilot.core.config import app_config
import brightai.logging

config = app_config()
brightai.logging.setup_logging(
    force_json=config.logging.log_json,
    root_logger_level=brightai.logging.LogLevels[config.logging.log_level],
    logger_levels={
        # Control third-party library log levels based on config
        "strands": brightai.logging.LogLevels[config.logging.strands_log_level],
        "strands.multiagent.a2a": brightai.logging.LogLevels[config.logging.strands_log_level],
        "strands.tools": brightai.logging.LogLevels[config.logging.strands_log_level],
        "strands.models": brightai.logging.LogLevels[config.logging.strands_log_level],
        "a2a": brightai.logging.LogLevels[config.logging.strands_log_level],
        # Keep these libraries at WARNING to reduce noise
        "boto3": brightai.logging.LogLevels.WARNING,
        "botocore": brightai.logging.LogLevels.WARNING,
        "urllib3": brightai.logging.LogLevels.WARNING,
        "httpcore": brightai.logging.LogLevels.WARNING,  # Reduce HTTP logging
        "httpx": brightai.logging.LogLevels.WARNING,
    },
)


def start() -> FastAPI:
    """Start the application with async initialization via startup events."""
    return initialize_app()


# Create the app instance for uvicorn import string
app = start()
