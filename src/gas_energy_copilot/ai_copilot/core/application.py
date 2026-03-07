from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from strands.multiagent.a2a import A2AServer
import structlog.stdlib

from gas_energy_copilot.ai_copilot.core.config import app_config
from gas_energy_copilot.ai_copilot.core.router import router
from gas_energy_copilot.ai_copilot.core.service_manager import initialize_expensive_services
from gas_energy_copilot.ai_copilot.middleware.logging import LoggingMiddleware
from gas_energy_copilot.ai_copilot.services.agent_service import AgentService

log = structlog.stdlib.get_logger()


def initialize_app() -> FastAPI:
    """Initialize and configure FastAPI application."""

    config = app_config()

    log.info("Starting application initialization...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifespan with proper error handling."""
        # Pre-initialize expensive services using the service manager
        try:
            log.info("Starting expensive services preloading...")
            await initialize_expensive_services(config)
            log.info("Expensive services preloading completed successfully")
        except Exception as e:
            log.error(f"Service preloading failed (will retry on first use): {e}")
            log.exception("Full exception details for service preloading failure:")

        log.info("Application initialization completed")

        yield  # Application runs here

        # Shutdown
        log.info("Application shutdown starting...")
        log.info("Application shutdown completed")

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

    agent = AgentService(config.agent, config).agent
    a2a_server = A2AServer(agent=agent, host=config.host, port=config.port, serve_at_root=True)
    a2a_app = a2a_server.to_fastapi_app()

    app.include_router(router)
    app.mount("/", a2a_app)

    return app
