import time
from typing import TypedDict

from asgi_correlation_id import correlation_id
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
import structlog.stdlib
from uvicorn.protocols.utils import get_path_with_query_string

from brightai.ai_copilot.core.config import app_config

config = app_config()
log = structlog.stdlib.get_logger()
access_log = structlog.stdlib.get_logger(__name__)


class AccessInfo(TypedDict, total=False):
    status_code: int
    start_time: float


class LoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
        self.config = config.logging

        self.excluded_path_prefixes: list[str] = sorted(
            self.config.log_access_excluded_path_prefixes, key=len, reverse=True
        )

    def _should_log_request(self, path: str, status_code: int) -> bool:
        """
        Determine if a request should be logged based on path and status code.

        Args:
            path: The request path
            status_code: The HTTP status code

        Returns:
            True if the request should be logged, False otherwise
        """
        # Check if path matches any excluded prefix
        path_matches = any(path.startswith(prefix) for prefix in self.excluded_path_prefixes)

        if not path_matches:
            # Path doesn't match exclusion list, always log
            return True

        # Path matches an excluded prefix
        if self.config.log_access_exclude_success_only:
            # Only exclude successful responses (2xx, 3xx)
            # Still log errors (4xx, 5xx) for debugging
            is_success = 200 <= status_code < 400
            return not is_success  # Log if NOT successful
        else:
            # Exclude ALL requests matching the path prefix
            return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # If the request is not an HTTP request, we don't need to do anything special
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=correlation_id.get())

        info = AccessInfo()

        # Inner send function
        async def inner_send(message):
            if message["type"] == "http.response.start":
                info["status_code"] = message["status"]
            await send(message)

        try:
            info["start_time"] = time.perf_counter_ns()
            await self.app(scope, receive, inner_send)
        except Exception as e:
            log.exception(
                "An unhandled exception was caught by last resort middleware",
                exception_class=e.__class__.__name__,
                exc_info=e,
                stack_info=config.debug,
            )
            info["status_code"] = 500
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred.",
                },
            )
            await response(scope, receive, send)
        finally:
            process_time = time.perf_counter_ns() - info["start_time"]
            client_host, client_port = scope["client"]
            http_method = scope["method"]
            http_version = scope["http_version"]
            url = get_path_with_query_string(scope)

            if self._should_log_request(scope["path"], info["status_code"]):
                # Recreate the Uvicorn access log format, but add all parameters as structured information
                access_log.info(
                    f"""{client_host}:{client_port} - "{http_method} {scope["path"]} HTTP/{http_version}" {info["status_code"]}""",  # noqa: E501
                    http={
                        "url": str(url),
                        "status_code": info["status_code"],
                        "method": http_method,
                        "request_id": correlation_id.get(),
                        "version": http_version,
                    },
                    network={"client": {"ip": client_host, "port": client_port}},
                    duration=process_time,
                )
            else:
                log.debug(
                    "Skipped request log",
                    http={
                        "url": str(url),
                        "status_code": info["status_code"],
                        "method": http_method,
                        "request_id": correlation_id.get(),
                        "version": http_version,
                    },
                    network={"client": {"ip": client_host, "port": client_port}},
                    duration=process_time,
                )
