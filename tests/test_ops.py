"""Regression tests for the empty-payload bug.

`/ops/metrics` returned `{"metrics": {}}` and `/ops/build` returned `{"build_info": {}}`
for the entire life of the project: both response models declared a single dict field, and
pydantic v2's `extra="ignore"` silently dropped every key that was passed. The test that
would have caught it existed and was failing on main — CI never ran pytest.
"""


def test_ops_metrics_is_populated(app_client):
    app_client.get("/health")
    payload = app_client.get("/ops/metrics").json()

    assert payload["window_size"] >= 1
    assert "routes" in payload and payload["routes"], "route breakdown must not be empty"
    assert payload["error_rate"] == 0.0
    assert payload["p95_ms"] >= 0.0
    route = payload["routes"][0]
    assert {"route", "requests", "errors", "avg_ms", "p50_ms", "p95_ms"} <= set(route)


def test_ops_build_reports_real_capabilities(app_client):
    payload = app_client.get("/ops/build").json()

    assert payload["external_cost_usd_per_request"] == 0.0
    assert payload["features"], "feature list must not be empty"
    # These are probed at runtime, not asserted by the README.
    for key in ("dense_backend_available", "spacy_available", "presidio_available", "langgraph_available"):
        assert isinstance(payload[key], bool)
    assert payload["retrieval_mode"] in {"lexical", "dense", "hybrid"}


def test_prometheus_endpoint_serves_metrics(app_client):
    response = app_client.get("/metrics")
    assert response.status_code == 200
    assert "http_request" in response.text
