"""Kalos portal - observability (docs/HARDENING.md, Phase 2).

Liveness/readiness endpoints for orchestrators and load balancers, a structured
request-log middleware, and a lightweight text `/metrics` feed. No external
dependency: request counters are in-process and the metrics format is
hand-rolled Prometheus text.

The access log emits ONE structured JSON line per request with only the
method, path, status, duration, and a request id - never request bodies,
headers, query strings, or tokens - so operational logging can never leak a
secret or client data.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("kalos.portal.access")

# Process start (monotonic) for uptime, and thread-safe request counters.
_START = time.monotonic()
_lock = threading.Lock()
_counts: Counter[str] = Counter()

# Probe endpoints an orchestrator hits constantly - counted, but not access-logged,
# so health checks don't drown the log.
_QUIET_PATHS = {"/healthz", "/readyz", "/metrics"}


def _record(status: int) -> None:
    with _lock:
        _counts["total"] += 1
        _counts[f"{status // 100}xx"] += 1


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request id, time the request, and emit one structured
    JSON access line. Logs method, path, status, and duration only - never
    bodies, headers, query strings, or tokens."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _record(500)
            self._emit(rid, request, 500, start, error=True)
            raise
        _record(response.status_code)
        if request.url.path not in _QUIET_PATHS:
            self._emit(rid, request, response.status_code, start)
        response.headers["X-Request-ID"] = rid
        return response

    @staticmethod
    def _emit(rid: str, request: Request, status: int, start: float, *, error: bool = False) -> None:
        line = {
            "rid": rid,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "dur_ms": round((time.perf_counter() - start) * 1000, 1),
        }
        if error:
            line["error"] = True
        log.info(json.dumps(line))


def metrics_text() -> str:
    """Prometheus-style text: process uptime + request counts by status class."""
    with _lock:
        snap = dict(_counts)
    uptime = time.monotonic() - _START
    lines = [
        "# HELP kalos_uptime_seconds Process uptime in seconds.",
        "# TYPE kalos_uptime_seconds gauge",
        f"kalos_uptime_seconds {uptime:.1f}",
        "# HELP kalos_requests_total Total HTTP requests handled.",
        "# TYPE kalos_requests_total counter",
        f"kalos_requests_total {snap.get('total', 0)}",
        "# HELP kalos_requests_by_class HTTP requests by status class.",
        "# TYPE kalos_requests_by_class counter",
    ]
    for cls in ("2xx", "3xx", "4xx", "5xx"):
        lines.append(f'kalos_requests_by_class{{class="{cls}"}} {snap.get(cls, 0)}')
    return "\n".join(lines) + "\n"
