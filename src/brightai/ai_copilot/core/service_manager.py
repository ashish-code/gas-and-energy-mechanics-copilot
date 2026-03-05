"""Service manager for handling expensive async initialization."""

import asyncio
import structlog.stdlib
from typing import Any, Dict, Optional, TypeVar

from brightai.ai_copilot.core.config import ApplicationConfig

log = structlog.stdlib.get_logger()

T = TypeVar("T")


class ServiceManager:
    """Manages async initialization of expensive services."""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialization_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def get_or_initialize(self, service_name: str, factory_func, *args, **kwargs) -> Any:
        """
        Get a service instance, initializing it asynchronously if not already done.

        Args:
            service_name: Unique name for the service
            factory_func: Async function that creates the service instance
            *args, **kwargs: Arguments to pass to the factory function

        Returns:
            The service instance
        """
        async with self._lock:
            if service_name in self._services:
                return self._services[service_name]

            # Check if initialization is already in progress
            if service_name in self._initialization_tasks:
                log.info(f"Waiting for {service_name} initialization to complete...")
                try:
                    service = await self._initialization_tasks[service_name]
                    return service
                except Exception as e:
                    log.error(f"Failed to initialize {service_name}: {e}")
                    # Remove failed task and try again
                    del self._initialization_tasks[service_name]

            # Start async initialization
            log.info(f"Starting async initialization of {service_name}...")
            task = asyncio.create_task(factory_func(*args, **kwargs))
            self._initialization_tasks[service_name] = task

            try:
                service = await task
                self._services[service_name] = service
                del self._initialization_tasks[service_name]
                log.info(f"Successfully initialized {service_name}")
                return service
            except Exception as e:
                log.error(f"Failed to initialize {service_name}: {e}")
                del self._initialization_tasks[service_name]
                raise

    async def preload_service(self, service_name: str, factory_func, *args, **kwargs):
        """
        Preload a service in the background without waiting for it.

        Args:
            service_name: Unique name for the service
            factory_func: Async function that creates the service instance
            *args, **kwargs: Arguments to pass to the factory function
        """
        log.info(f"preload_service called for {service_name}")
        if service_name not in self._services and service_name not in self._initialization_tasks:
            log.info(f"Preloading {service_name} in background...")
            try:
                asyncio.create_task(self.get_or_initialize(service_name, factory_func, *args, **kwargs))
                log.info(f"Background task created for {service_name}")
            except Exception as e:
                log.error(f"Failed to create background task for {service_name}: {e}")
                log.exception("Background task creation exception details:")
        else:
            log.info(f"Service {service_name} already exists or is being initialized")

    def get_service_sync(self, service_name: str) -> Optional[Any]:
        """
        Get a service instance synchronously if it's already initialized.

        Args:
            service_name: Unique name for the service

        Returns:
            The service instance if available, None otherwise
        """
        return self._services.get(service_name)

    async def clear_service(self, service_name: str):
        """Clear a specific service from the cache."""
        async with self._lock:
            if service_name in self._services:
                del self._services[service_name]
                log.info(f"Cleared {service_name} from cache")

    async def clear_all_services(self):
        """Clear all services from the cache."""
        async with self._lock:
            self._services.clear()
            # Cancel any pending initialization tasks
            for task in self._initialization_tasks.values():
                if not task.done():
                    task.cancel()
            self._initialization_tasks.clear()
            log.info("Cleared all services from cache")


# Global service manager instance
_service_manager = ServiceManager()


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance."""
    return _service_manager


async def initialize_rag_service(config: ApplicationConfig):
    """Factory function to create and initialize RAG service."""
    from brightai.ai_copilot.services.rag_service import RAGService

    log.info("Creating RAG service instance")
    rag_service = RAGService(config.rag)

    # Run initialization in executor since it's CPU-bound
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rag_service.initialize)

    return rag_service


async def initialize_expensive_services(config: ApplicationConfig):
    """Initialize expensive services asynchronously."""
    log.info("initialize_expensive_services called")
    service_manager = get_service_manager()
    log.info("Service manager retrieved")

    # Preload RAG service if enabled
    if config.rag.enabled:
        log.info("RAG is enabled, preloading RAG service...")
        await service_manager.preload_service("rag", initialize_rag_service, config)

    log.info("Expensive services preloading initiated")
