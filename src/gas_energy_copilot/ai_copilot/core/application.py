from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import structlog.stdlib

from gas_energy_copilot.ai_copilot.core.config import app_config
from gas_energy_copilot.ai_copilot.core.router import router
from gas_energy_copilot.ai_copilot.middleware.logging import LoggingMiddleware

log = structlog.stdlib.get_logger()


def initialize_app() -> FastAPI:
    """Initialize and configure the FastAPI application."""

    config = app_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("application_startup")
        yield
        log.info("application_shutdown")

    app = FastAPI(
        title=config.app_name,
        docs_url="/docs" if config.api.serve_docs_enabled else None,
        redoc_url="/redoc" if config.api.serve_docs_enabled else None,
        openapi_url="/openapi.json" if config.api.serve_docs_enabled else None,
        debug=config.debug,
        lifespan=lifespan,
    )

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
    )

    app.include_router(router)

    return app
