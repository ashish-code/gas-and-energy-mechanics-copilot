"""Process-global minimum-interval rate limiter for Bedrock calls.

This account throttles around ~1 req/s. Under sustained batch load, *hitting* the throttle
triggers exponential backoff that is far slower than simply pacing under it. A single global
gate that spaces calls by a minimum interval keeps us just under the limit, so we avoid
ThrottlingException storms entirely — both during the index build and at query time.

`BEDROCK_MIN_INTERVAL` (env, seconds) tunes the spacing; default 1.2s gives margin under the
account's variable limit (0.9s occasionally still tripped throttling during the build).
"""

from __future__ import annotations

import os
import threading
import time

_MIN_INTERVAL = float(os.environ.get("BEDROCK_MIN_INTERVAL", "1.2"))
_lock = threading.Lock()
_next_allowed = 0.0


def throttle() -> None:
    """Block until the global minimum interval since the previous call has elapsed."""
    global _next_allowed
    if _MIN_INTERVAL <= 0:
        return
    with _lock:
        now = time.monotonic()
        wait = _next_allowed - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _next_allowed = now + _MIN_INTERVAL
