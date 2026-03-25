from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings


ROLE = {
    "title": "AI Engineer",
    "description": (
        "We are hiring an AI Engineer to build retrieval-backed applications using Python, FastAPI, vector search, "
        "evaluation pipelines, Docker, cloud deployment, caching, and production APIs. Strong experience with machine learning, "
        "RAG, observability, and shipping end-to-end systems is preferred."
    ),
}

CANDIDATES = [
    {
        "filename": "akila_resume.txt",
        "text": "Akila built FastAPI services, retrieval systems, Dockerized ML apps, CI pipelines, cloud deployments, and evaluation dashboards using Python and RAG.",
        "expected_strengths": ["FastAPI", "retrieval", "Docker", "cloud"],
    },
    {
        "filename": "esha_resume.txt",
        "text": "Esha has machine learning coursework, experimentation, Python, notebooks, and a Streamlit demo with some model evaluation work.",
        "expected_strengths": ["machine learning", "Python", "evaluation"],
    },
    {
        "filename": "jaxon_resume.txt",
        "text": "Jaxon has entry-level Python, HTML, and class projects with limited backend or production ML experience.",
        "expected_strengths": ["Python"],
    },
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, round((pct / 100) * (len(values) - 1))))
    return values[idx]


def timed(fn):
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--assert', dest='do_assert', action='store_true')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['DATA_DIR'] = tmpdir
        os.environ['VECTORSTORE_PATH'] = str(Path(tmpdir) / 'vectorstore')
        get_settings.cache_clear()
        from app.main import app

        role_latencies = []
        upload_latencies = []
        evaluate_latencies = []
        copilot_latencies = []

        with TestClient(app) as client:
            role_resp, ms = timed(lambda: client.post('/roles/text', json=ROLE))
            role_latencies.append(ms)
            role_resp.raise_for_status()
            role = role_resp.json()
            role_id = role['id']

            files = [('resumes', (candidate['filename'], candidate['text'].encode('utf-8'), 'text/plain')) for candidate in CANDIDATES]
            upload_resp, ms = timed(lambda: client.post('/candidates/upload', files=files))
            upload_latencies.append(ms)
            upload_resp.raise_for_status()
            upload_payload = upload_resp.json()
            candidate_ids = [item['id'] for item in upload_payload['candidates']]

            for _ in range(20):
                _, ms = timed(lambda: client.post(f'/roles/{role_id}/evaluate', json={'candidate_ids': candidate_ids, 'top_k_per_requirement': 2}))
                evaluate_latencies.append(ms)
            eval_resp = client.post(f'/roles/{role_id}/evaluate', json={'candidate_ids': candidate_ids, 'top_k_per_requirement': 2})
            eval_resp.raise_for_status()
            evaluation = eval_resp.json()
            ranked_candidates = evaluation['candidates']
            top_candidate = ranked_candidates[0]['candidate_name'] if ranked_candidates else ''

            question = 'Based on the evaluation results, who is the strongest candidate for this role?'
            for _ in range(20):
                _, ms = timed(lambda: client.post('/copilot/query', json={'question': question, 'role_id': role_id, 'candidate_ids': candidate_ids, 'top_k': 8}))
                copilot_latencies.append(ms)
            copilot_resp = client.post('/copilot/query', json={'question': question, 'role_id': role_id, 'candidate_ids': candidate_ids, 'top_k': 8})
            copilot_resp.raise_for_status()
            copilot_answer = copilot_resp.json()['answer']

            metrics_resp = client.get('/ops/metrics')
            metrics_resp.raise_for_status()
            ops_metrics = metrics_resp.json()

        metrics = {
            'benchmark_context': 'in-process FastAPI TestClient on local CPU with no external API calls',
            'slo_targets': {
                'evaluate_p95_ms': 1500,
                'copilot_p95_ms': 1500,
                'external_cost_usd_per_request': 0.0,
            },
            'latency_ms': {
                'role_create_p95': round(percentile(role_latencies, 95), 2),
                'candidate_upload_p95': round(percentile(upload_latencies, 95), 2),
                'evaluate_p95': round(percentile(evaluate_latencies, 95), 2),
                'copilot_p95': round(percentile(copilot_latencies, 95), 2),
            },
            'quality': {
                'top1_eval_accuracy': 1.0 if top_candidate.lower().startswith('akila') else 0.0,
                'copilot_ranking_consistency': 1.0 if top_candidate and top_candidate in copilot_answer else 0.0,
                'candidate_count': len(candidate_ids),
                'requirements_per_role': len(role['requirements']),
            },
            'cost': {
                'external_api_cost_usd_per_request': 0.0,
                'hosting_note': 'Excludes infrastructure hosting cost; local CPU benchmark only.',
            },
            'ops_metrics_snapshot': ops_metrics,
        }

        out_dir = Path('docs')
        out_dir.mkdir(exist_ok=True)
        (out_dir / 'benchmark_results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        (out_dir / 'benchmark.md').write_text(
            "\n".join([
                '# Benchmark summary',
                '',
                f"- Evaluate p95: {metrics['latency_ms']['evaluate_p95']} ms",
                f"- Copilot p95: {metrics['latency_ms']['copilot_p95']} ms",
                f"- Upload p95: {metrics['latency_ms']['candidate_upload_p95']} ms",
                f"- Top-1 evaluation accuracy: {metrics['quality']['top1_eval_accuracy']}",
                f"- Copilot ranking consistency: {metrics['quality']['copilot_ranking_consistency']}",
                f"- External API cost per request: ${metrics['cost']['external_api_cost_usd_per_request']:.3f}",
            ]),
            encoding='utf-8',
        )
        print(json.dumps(metrics, indent=2))

        if args.do_assert:
            assert metrics['latency_ms']['evaluate_p95'] < 1500
            assert metrics['latency_ms']['copilot_p95'] < 1500
            assert metrics['quality']['top1_eval_accuracy'] >= 1.0
            assert metrics['quality']['copilot_ranking_consistency'] >= 1.0


if __name__ == '__main__':
    main()
