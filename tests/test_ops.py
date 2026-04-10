from fastapi.testclient import TestClient

from app.main import app


def test_ops_metrics_and_build():
    with TestClient(app) as client:
        health = client.get('/health')
        assert health.status_code == 200

        metrics = client.get('/ops/metrics')
        assert metrics.status_code == 200
        payload = metrics.json()
        assert 'routes' in payload
        assert payload['window_size'] >= 1

        build = client.get('/ops/build')
        assert build.status_code == 200
        build_payload = build.json()
        assert build_payload['external_cost_usd_per_request'] == 0.0
        assert 'Docker + CI' in build_payload['features']
