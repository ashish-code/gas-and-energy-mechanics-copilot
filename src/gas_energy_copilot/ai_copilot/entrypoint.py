from fastapi import FastAPI

from gas_energy_copilot.ai_copilot.core.application import initialize_app
from gas_energy_copilot.ai_copilot.core.config import app_config
import gas_energy_copilot.logging

config = app_config()
gas_energy_copilot.logging.setup_logging(
    force_json=config.logging.log_json,
    root_logger_level=gas_energy_copilot.logging.LogLevels[config.logging.log_level],
    logger_levels={
        # Control third-party library log levels based on config
        "strands": gas_energy_copilot.logging.LogLevels[config.logging.strands_log_level],
        "strands.multiagent.a2a": gas_energy_copilot.logging.LogLevels[config.logging.strands_log_level],
        "strands.tools": gas_energy_copilot.logging.LogLevels[config.logging.strands_log_level],
        "strands.models": gas_energy_copilot.logging.LogLevels[config.logging.strands_log_level],
        "a2a": gas_energy_copilot.logging.LogLevels[config.logging.strands_log_level],
        # Keep these libraries at WARNING to reduce noise
        "boto3": gas_energy_copilot.logging.LogLevels.WARNING,
        "botocore": gas_energy_copilot.logging.LogLevels.WARNING,
        "urllib3": gas_energy_copilot.logging.LogLevels.WARNING,
        "httpcore": gas_energy_copilot.logging.LogLevels.WARNING,  # Reduce HTTP logging
        "httpx": gas_energy_copilot.logging.LogLevels.WARNING,
    },
)


def start() -> FastAPI:
    """Start the application with async initialization via startup events."""
    return initialize_app()


# Create the app instance for uvicorn import string
app = start()
