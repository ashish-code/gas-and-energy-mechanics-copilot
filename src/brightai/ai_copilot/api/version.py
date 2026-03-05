from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/",
    summary="Get the version of the Kodiak AI Copilot service",
    description="Return the build version of the currently running service",
    name="version",
)
@router.get(
    "",
    summary="Get the version of the Kodiak AI Copilot service (without trailing slash)",
    description="Return the build version of the currently running service",
    name="version_no_slash",
    include_in_schema=False,
)
async def get_version() -> str:
    """Get the application version."""
    from brightai.ai_copilot.api.__version__ import __version__

    return __version__
