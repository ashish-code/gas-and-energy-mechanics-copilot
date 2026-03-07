from fastapi import APIRouter
import structlog.stdlib

from gas_energy_copilot.ai_copilot.api import debug, health, version
from gas_energy_copilot.ai_copilot.api.v1.endpoints import api_router
from gas_energy_copilot.ai_copilot.core.config import app_config

config = app_config()
log = structlog.stdlib.get_logger()

router = APIRouter()

# System-level Endpoints
router.include_router(health.router, prefix="/health", tags=["system"])
router.include_router(version.router, prefix="/version", tags=["system"])
if config.debug:
    router.include_router(debug.router, prefix="/debug", tags=["system"])

# Versioned API Endpoints
router.include_router(api_router, prefix=f"/{config.api.api_version}")


@router.get(
    "/",
    summary="Get the root message",
    description="Return the root message from configuration",
    name="root",
)
async def root() -> dict[str, str]:
    """Get the root message."""
    return {"message": config.api.root_message}
