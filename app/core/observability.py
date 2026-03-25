from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from statistics import mean
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


class RequestMetricsStore:
    def __init__(self, *, window_size: int = 1000) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=window_size)

    def record(self, *, path: str, method: str, status_code: int, duration_ms: float) -> None:
        self._records.append(
            {
                "path": path,
                "method": method,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
            }
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._records)

    def summary(self) -> dict[str, Any]:
        records = self.snapshot()
        durations = [item["duration_ms"] for item in records]
        error_count = sum(1 for item in records if item["status_code"] >= 400)
        by_route: dict[str, list[dict[str, Any]]] = {}
        for item in records:
            key = f'{item["method"]} {item["path"]}'
            by_route.setdefault(key, []).append(item)

        route_metrics = []
        for route, items in sorted(by_route.items()):
            route_durations = [item["duration_ms"] for item in items]
            route_metrics.append(
                {
                    "route": route,
                    "requests": len(items),
                    "errors": sum(1 for item in items if item["status_code"] >= 400),
                    "avg_ms": round(mean(route_durations), 2) if route_durations else 0.0,
                    "p50_ms": round(_percentile(route_durations, 50), 2) if route_durations else 0.0,
                    "p95_ms": round(_percentile(route_durations, 95), 2) if route_durations else 0.0,
                }
            )

        return {
            "window_size": len(records),
            "error_rate": round(error_count / len(records), 4) if records else 0.0,
            "avg_ms": round(mean(durations), 2) if durations else 0.0,
            "p50_ms": round(_percentile(durations, 50), 2) if durations else 0.0,
            "p95_ms": round(_percentile(durations, 95), 2) if durations else 0.0,
            "routes": route_metrics,
        }


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, metrics_store: RequestMetricsStore) -> None:
        super().__init__(app)
        self._metrics_store = metrics_store

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            self._metrics_store.record(
                path=request.url.path,
                method=request.method,
                status_code=500,
                duration_ms=duration_ms,
            )
            logger.exception(
                "Unhandled request failure",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        self._metrics_store.record(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
