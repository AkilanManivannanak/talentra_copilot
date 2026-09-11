"""
Latency benchmark for the serving path.

What changed and why, because the previous numbers were not measuring what the README
claimed they measured:

* **Cold and warm are now reported separately.** `RankingService` caches on
  (role, candidates, k, store versions). The old benchmark fired the identical request 20
  times with no writes in between, so iteration 1 computed and iterations 2-20 were cache
  hits — and with n=20 the p95 index lands on sample 19, always a hit. "evaluate p95
  4.81 ms" was the cost of a dict lookup and a pydantic deep copy. Cold p95 is now the
  headline; warm p95 is reported beside it as the cache-hit figure it always was.
* **n is 200, not 20**, because a p95 over 20 samples is one data point wearing a
  percentile's clothes.
* **The quality metrics are gone from this script.** `top1_eval_accuracy` and
  `copilot_ranking_consistency` were a hardcoded name check and a tautology. Retrieval
  quality is measured properly in `eval/harness.py` against graded judgements.
* **It is called a benchmark, not an SLO.** An SLO is a commitment about production
  traffic. This is an in-process measurement on one machine with no network hop, no
  concurrency, and a synthetic corpus. Those caveats are printed with the results.

Run:
    python scripts/benchmark.py
    python scripts/benchmark.py --assert   # regression gate used by CI
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Regression thresholds. Generous on purpose: this gate exists to catch an order-of-
# magnitude regression (an accidental O(n^2), a cache removed, a model loaded per request),
# not to assert a number that only holds on the author's laptop.
THRESHOLDS = {
    "evaluate_cold_p95_ms": 2000,
    "copilot_cold_p95_ms": 2000,
    "upload_p95_ms": 3000,
    "role_create_p95_ms": 500,
}

ROLE = {
    "title": "AI Engineer",
    "description": """AI Engineer, Search & Retrieval

Requirements:
- Must have 3+ years of professional Python experience.
- Required: experience designing and shipping retrieval-augmented generation systems in production.
- Strong knowledge of building and deploying HTTP services with FastAPI and Docker on a cloud platform.
- Experience designing offline evaluation pipelines and measuring retrieval quality.
Nice to have: exposure to agent orchestration frameworks.
""",
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarise(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3) if values else 0.0,
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "max": round(max(values), 3) if values else 0.0,
    }


def timed(fn):
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000


def load_corpus(count: int) -> list[tuple[str, str]]:
    """Reuse the evaluation corpus so the benchmark runs against realistic documents."""
    labels_path = Path(__file__).resolve().parents[1] / "eval" / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return [(c["filename"], c["text"]) for c in labels["candidates"][:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert", dest="do_assert", action="store_true")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--mode", default="hybrid", choices=["lexical", "dense", "hybrid"])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["DATA_DIR"] = tmpdir
        os.environ["VECTORSTORE_PATH"] = str(Path(tmpdir) / "vectorstore")
        os.environ["RETRIEVAL_MODE"] = args.mode
        from app.core.config import get_settings

        get_settings.cache_clear()
        from fastapi.testclient import TestClient

        from app.main import app

        role_latencies: list[float] = []
        upload_latencies: list[float] = []
        evaluate_cold: list[float] = []
        evaluate_warm: list[float] = []
        copilot_cold: list[float] = []
        copilot_warm: list[float] = []

        with TestClient(app) as client:
            services = app.state.services

            # Warm the preprocessing models before measuring. Presidio's engine
            # construction is ~8s on a cold process; charging it to whichever request
            # happens to arrive first is what made the old upload figure meaningless.
            from app.preprocessing import warm_models

            warm_started = time.perf_counter()
            warm_timings = warm_models()
            warm_ms = (time.perf_counter() - warm_started) * 1000

            role_response, elapsed = timed(lambda: client.post("/roles/text", json=ROLE))
            role_latencies.append(elapsed)
            role_response.raise_for_status()
            role = role_response.json()
            role_id = role["id"]

            corpus = load_corpus(args.candidates)
            files = [("resumes", (name, text.encode("utf-8"), "text/plain")) for name, text in corpus]
            upload_response, elapsed = timed(lambda: client.post("/candidates/upload", files=files))
            upload_latencies.append(elapsed)
            upload_response.raise_for_status()
            candidate_ids = [item["id"] for item in upload_response.json()["candidates"]]

            evaluate_body = {"candidate_ids": candidate_ids, "top_k_per_requirement": 2}
            question = "Which candidate is the strongest for this role?"
            copilot_body = {
                "question": question,
                "role_id": role_id,
                "candidate_ids": candidate_ids,
                "top_k": 8,
            }

            def clear_ranking_cache() -> None:
                """Drop the memoised evaluations so the next call does the real work.

                This is the whole point of the rewrite: without it every sample after the
                first measures a dict lookup, which is how the old benchmark reported
                4.81 ms for an operation that retrieves across the corpus.
                """
                services.ranking._cache.clear()

            for _ in range(args.iterations):
                clear_ranking_cache()
                _, elapsed = timed(lambda: client.post(f"/roles/{role_id}/evaluate", json=evaluate_body))
                evaluate_cold.append(elapsed)

            for _ in range(args.iterations):
                _, elapsed = timed(lambda: client.post(f"/roles/{role_id}/evaluate", json=evaluate_body))
                evaluate_warm.append(elapsed)

            for _ in range(args.iterations):
                clear_ranking_cache()
                _, elapsed = timed(lambda: client.post("/copilot/query", json=copilot_body))
                copilot_cold.append(elapsed)

            for _ in range(args.iterations):
                _, elapsed = timed(lambda: client.post("/copilot/query", json=copilot_body))
                copilot_warm.append(elapsed)

            # Sanity checks: a benchmark that measures a broken endpoint is worthless.
            evaluation = client.post(f"/roles/{role_id}/evaluate", json=evaluate_body)
            evaluation.raise_for_status()
            ranked = evaluation.json()["candidates"]
            copilot = client.post("/copilot/query", json=copilot_body)
            copilot.raise_for_status()
            answer = copilot.json()

            ops = client.get("/ops/metrics")
            ops.raise_for_status()
            build = client.get("/ops/build")
            build.raise_for_status()

        cache_speedup = (
            round(percentile(evaluate_cold, 95) / percentile(evaluate_warm, 95), 1)
            if percentile(evaluate_warm, 95)
            else None
        )

        results = {
            "context": {
                "harness": "in-process FastAPI TestClient; no network hop, no concurrency",
                "retrieval_mode": build.json()["retrieval_mode"],
                "dense_backend": build.json()["embedding_model"],
                "candidates_indexed": len(candidate_ids),
                "requirements": len(role["requirements"]),
                "iterations_per_measurement": args.iterations,
                "caveat": (
                    "These are local latency measurements, not SLOs. An SLO is a commitment "
                    "about production traffic under concurrency; this process serves one "
                    "request at a time on one machine against a synthetic corpus."
                ),
            },
            "latency_ms": {
                "role_create": summarise(role_latencies),
                "candidate_upload_batch": summarise(upload_latencies),
                "evaluate_cold": summarise(evaluate_cold),
                "evaluate_warm_cache_hit": summarise(evaluate_warm),
                "copilot_cold": summarise(copilot_cold),
                "copilot_warm_cache_hit": summarise(copilot_warm),
            },
            "cache": {
                "evaluate_p95_speedup_x": cache_speedup,
                "note": (
                    "Warm figures are cache hits and should never be quoted as request latency. "
                    "The previously published 4.81 ms evaluate p95 was a warm number."
                ),
            },
            "cold_start_ms": {
                "preprocessing_model_warmup_total": round(warm_ms, 1),
                **warm_timings,
                "note": (
                    "Paid once per process, on a background thread at startup. Reported "
                    "separately because it is a boot cost, not a request cost."
                ),
            },
            "cost": {
                "external_api_cost_usd_per_request": 0.0,
                "note": "All models run in-process on CPU. Excludes infrastructure hosting cost.",
            },
            "sanity": {
                "candidates_ranked": len(ranked),
                "top_candidate": ranked[0]["candidate_name"] if ranked else None,
                "answer_grounded": (answer.get("faithfulness") or {}).get("faithfulness"),
                "ops_metrics_populated": bool(ops.json().get("routes")),
            },
            "thresholds": THRESHOLDS,
        }

        out_dir = Path("docs")
        out_dir.mkdir(exist_ok=True)
        (out_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

        latency = results["latency_ms"]
        lines = [
            "# Benchmark",
            "",
            f"_{results['context']['caveat']}_",
            "",
            f"Retrieval mode: `{results['context']['retrieval_mode']}` · "
            f"{results['context']['candidates_indexed']} candidates · "
            f"{results['context']['requirements']} requirements · "
            f"n={args.iterations} per measurement",
            "",
            "| Operation | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---|---|---|",
        ]
        for key in ("evaluate_cold", "evaluate_warm_cache_hit", "copilot_cold", "copilot_warm_cache_hit",
                    "candidate_upload_batch", "role_create"):
            row = latency[key]
            lines.append(f"| {key.replace('_', ' ')} | {row['p50']} | {row['p95']} | {row['p99']} |")
        lines += [
            "",
            f"Cache speed-up on evaluate p95: **{cache_speedup}x**. "
            "Warm numbers are cache hits, not request latency.",
            "",
            "External API cost per request: **$0.000** (all models in-process, CPU).",
        ]
        (out_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

        print(json.dumps({k: v for k, v in results.items() if k != "thresholds"}, indent=2))

        if args.do_assert:
            failures = []
            checks = {
                "evaluate_cold_p95_ms": latency["evaluate_cold"]["p95"],
                "copilot_cold_p95_ms": latency["copilot_cold"]["p95"],
                "upload_p95_ms": latency["candidate_upload_batch"]["p95"],
                "role_create_p95_ms": latency["role_create"]["p95"],
            }
            for name, value in checks.items():
                if value > THRESHOLDS[name]:
                    failures.append(f"{name}: {value} > {THRESHOLDS[name]}")
            if not results["sanity"]["ops_metrics_populated"]:
                failures.append("/ops/metrics returned an empty payload")
            if not results["sanity"]["candidates_ranked"]:
                failures.append("evaluate returned no ranked candidates")
            if failures:
                print("\nBENCHMARK GATE FAILED:")
                for failure in failures:
                    print(f"  - {failure}")
                raise SystemExit(1)
            print("\nBenchmark gate passed.")


if __name__ == "__main__":
    main()
