"""
app/core/prometheus.py

Prometheus metrics integration for Talentra Copilot.
Exposes GET /metrics — standard Prometheus scrape endpoint.

Usage — add to app/main.py after `app = FastAPI(...)`:

    from app.core.prometheus import setup_prometheus
    setup_prometheus(app)

That's it. /metrics will be live.
"""

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def setup_prometheus(app: FastAPI) -> None:
    """
    Instrument the FastAPI app with Prometheus metrics.

    Adds GET /metrics scrape endpoint exposing:
      - http_requests_total          (counter, by route/method/status)
      - http_request_duration_seconds (histogram, p50/p95/p99)
      - http_requests_in_progress    (gauge)

    Compatible with Prometheus server, Grafana Cloud, Render metrics.
    Complements the existing /ops/metrics custom in-memory store.
    """
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/docs", "/openapi.json"],
        inprogress_name="talentra_http_requests_inprogress",
        inprogress_labels=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
