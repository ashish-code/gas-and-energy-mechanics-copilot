from fastapi import APIRouter, HTTPException, status
import structlog.stdlib

from gas_energy_copilot.ai_copilot.core.config import app_config

config = app_config()
log = structlog.stdlib.get_logger()

router = APIRouter()


@router.get(
    "/",
    summary="Health check endpoint",
    description="Check if the service is healthy",
    name="health",
)
@router.get(
    "",
    summary="Health check endpoint (without trailing slash)",
    description="Check if the service is healthy",
    name="health_no_slash",
    include_in_schema=False,
)
async def health_check():
    """Health check endpoint for the API."""
    try:
        # Check if service is healthy
        is_healthy = True

        if not is_healthy:
            log.warning("Health check failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unhealthy - health checks not responding",
            )

        return {
            "status": "healthy",
            "service": config.service_name,
        }
    except Exception as e:
        log.error(f"Health check failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Health check failed: {str(e)}")
