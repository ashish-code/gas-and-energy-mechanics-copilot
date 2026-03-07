import attrs
from fastapi import APIRouter
import structlog.stdlib

from gas_energy_copilot.ai_copilot.core.config import app_config

config = app_config()
log = structlog.stdlib.get_logger()

router = APIRouter()


@router.get(
    "/settings",
    summary="Debug settings endpoint",
    description="Displays settings when debug mode is enabled",
    name="debug_settings",
)
async def get_settings():
    """Debug endpoint which displays application settings"""
    settings = attrs.asdict(config)
    return settings
