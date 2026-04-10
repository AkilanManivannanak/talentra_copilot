from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestMetrics:
    """In-memory latency store per route."""
    def __init__(self):
        self._latencies: dict[str, list[float]] = defaultdict(list)

    def record(self, route: str, ms: float) -> None:
        self._latencies[route].append(ms)

    def summary(self) -> dict:
        out = {}
        for route, lats in self._latencies.items():
            if not lats:
                continue
            s = sorted(lats)
            n = len(s)
            out[route] = {
                "count": n,
                "p50_ms": round(s[n // 2], 2),
                "p95_ms": round(s[min(n - 1, int(n * 0.95))], 2),
                "p99_ms": round(s[min(n - 1, int(n * 0.99))], 2),
            }
        return out


metrics = RequestMetrics()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        route = request.url.path
        metrics.record(route, elapsed_ms)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
