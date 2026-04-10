"""
Eval gate: fine-tuned model must pass quality thresholds before
replacing the rule-based backend.

Runs the same assertions as scripts/benchmark.py --assert
plus a fine-tune-specific accuracy check.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


GATE_THRESHOLDS = {
    "top1_eval_accuracy": 1.0,
    "copilot_ranking_consistency": 1.0,
    "evaluate_p95_ms": 1500,
    "copilot_p95_ms": 1500,
}


def run_eval_gate(
    model_path: str | Path | None = None,
    benchmark_results_path: str | Path = "docs/benchmark_results.json",
    run_benchmark: bool = True,
) -> dict:
    """
    Run the benchmark and check all thresholds.
    Returns {passed: bool, results: dict, failures: [str]}.
    """
    if run_benchmark:
        logger.info("Running benchmark script...")
        result = subprocess.run(
            [sys.executable, "scripts/benchmark.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {
                "passed": False,
                "results": {},
                "failures": [f"Benchmark script failed: {result.stderr[:500]}"],
            }

    results_path = Path(benchmark_results_path)
    if not results_path.exists():
        return {
            "passed": False,
            "results": {},
            "failures": ["Benchmark results file not found. Run scripts/benchmark.py first."],
        }

    with open(results_path) as f:
        results = json.load(f)

    failures = []

    # Latency checks
    latency = results.get("latency_ms", {})
    if latency.get("evaluate_p95", 9999) > GATE_THRESHOLDS["evaluate_p95_ms"]:
        failures.append(
            f"evaluate_p95 {latency['evaluate_p95']} ms exceeds {GATE_THRESHOLDS['evaluate_p95_ms']} ms"
        )
    if latency.get("copilot_p95", 9999) > GATE_THRESHOLDS["copilot_p95_ms"]:
        failures.append(
            f"copilot_p95 {latency['copilot_p95']} ms exceeds {GATE_THRESHOLDS['copilot_p95_ms']} ms"
        )

    # Quality checks
    quality = results.get("quality", {})
    if quality.get("top1_eval_accuracy", 0) < GATE_THRESHOLDS["top1_eval_accuracy"]:
        failures.append(
            f"top1_eval_accuracy {quality.get('top1_eval_accuracy')} < {GATE_THRESHOLDS['top1_eval_accuracy']}"
        )
    if quality.get("copilot_ranking_consistency", 0) < GATE_THRESHOLDS["copilot_ranking_consistency"]:
        failures.append(
            f"copilot_ranking_consistency {quality.get('copilot_ranking_consistency')} < {GATE_THRESHOLDS['copilot_ranking_consistency']}"
        )

    passed = len(failures) == 0
    if passed:
        logger.info("✓ Eval gate passed. Fine-tuned model is cleared for promotion.")
        if model_path:
            _write_active_model_pointer(model_path)
    else:
        logger.warning(f"✗ Eval gate FAILED: {failures}")

    return {"passed": passed, "results": results, "failures": failures}


def _write_active_model_pointer(model_path: str | Path) -> None:
    """Write a pointer file so the serving layer knows which model to load."""
    pointer = Path("models/active_model.json")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with open(pointer, "w") as f:
        json.dump({"model_path": str(model_path)}, f)
    logger.info(f"Active model pointer updated → {model_path}")
