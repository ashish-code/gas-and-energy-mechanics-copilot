from fastapi import FastAPI

from gas_energy_copilot.ai_copilot.core.application import initialize_app
from gas_energy_copilot.ai_copilot.core.config import app_config
import gas_energy_copilot.logging

config = app_config()
gas_energy_copilot.logging.setup_logging(
    force_json=config.logging.log_json,
    root_logger_level=gas_energy_copilot.logging.LogLevels[config.logging.log_level],
    logger_levels={
        "boto3": gas_energy_copilot.logging.LogLevels.WARNING,
        "botocore": gas_energy_copilot.logging.LogLevels.WARNING,
        "urllib3": gas_energy_copilot.logging.LogLevels.WARNING,
        "httpcore": gas_energy_copilot.logging.LogLevels.WARNING,
        "httpx": gas_energy_copilot.logging.LogLevels.WARNING,
        "crewai": gas_energy_copilot.logging.LogLevels.WARNING,
        "litellm": gas_energy_copilot.logging.LogLevels.WARNING,
    },
)


def start() -> FastAPI:
    return initialize_app()


app = start()
