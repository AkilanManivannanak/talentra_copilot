"""
Promotion gate for a fine-tuned model.

The previous implementation was a no-op with respect to the thing it was gating. It
shelled out to `scripts/benchmark.py`, which builds the FastAPI app and exercises the
rule-based lexical pipeline — the adapter at `model_path` was never loaded, never
inferenced, and never compared against anything. On "pass" it wrote
`models/active_model.json`, a pointer file that nothing in the repository reads. So a
LoRA run could be "promoted to active" without a single token passing through it.

This version requires a model in the loop:

* Both the base model and the adapter answer the same held-out prompts.
* Each answer is scored for **groundedness** against the evidence in the prompt, using
  the same `FaithfulnessScorer` the serving path uses.
* The adapter is promoted only if it beats the base model by a margin, and only if the
  serving-path benchmark still passes its regression thresholds.

If transformers/peft are not installed, or no adapter path is supplied, the gate refuses
to promote and says so. It does not silently "pass".
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)

# Minimum groundedness improvement over the base model, in absolute points, before an
# adapter is worth promoting. Below this the difference is noise on a set this size.
MIN_GROUNDEDNESS_GAIN = 0.05
MIN_ABSOLUTE_GROUNDEDNESS = 0.60


def _held_out_prompts(limit: int = 12) -> list[dict[str, Any]]:
    """Prompts built from the evaluation corpus, with the evidence the answer must stay
    inside. Deliberately not the same examples the SFT data was generated from."""
    labels_path = Path(__file__).resolve().parents[2] / "eval" / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    prompts = []
    for role in labels["roles"]:
        for requirement in role["requirements"][:2]:
            for candidate_id in role["candidate_ids"][-2:]:
                candidate = next(c for c in labels["candidates"] if c["id"] == candidate_id)
                prompts.append(
                    {
                        "question": f"What evidence does this candidate have for: {requirement['text']}?",
                        "evidence": candidate["text"],
                        "candidate_name": candidate["name"],
                    }
                )
    return prompts[:limit]


def _score_groundedness(answers: Sequence[str], prompts: Sequence[dict]) -> float:
    from app.models.schemas import Evidence
    from app.services.faithfulness import FaithfulnessScorer

    scorer = FaithfulnessScorer()
    scores = []
    for answer, prompt in zip(answers, prompts, strict=False):
        citation = Evidence(
            document_id="eval", filename="eval.txt", entity_id="eval",
            entity_name=prompt["candidate_name"], snippet=prompt["evidence"], score=1.0,
        )
        scores.append(scorer.score(answer=answer, citations=[citation]).faithfulness)
    return round(mean(scores), 4) if scores else 0.0


def _generate(model_path: str | None, base_model: str, prompts: Sequence[dict]) -> list[str] | None:
    """Run the prompts through base or adapter. Returns None when deps are unavailable."""
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        logger.warning("Generation dependencies unavailable: %s", exc)
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path or base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        if model_path:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, model_path)
        model.eval()
    except Exception as exc:
        logger.warning("Could not load model (%s): %s", model_path or base_model, exc)
        return None

    answers = []
    for prompt in prompts:
        text = (
            "You are an evidence-grounded hiring assistant. Answer using only the evidence.\n\n"
            f"Evidence: {prompt['evidence']}\n\nQuestion: {prompt['question']}\nAnswer:"
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        answers.append(
            tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        )
    return answers


def run_eval_gate(
    model_path: str | Path | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    benchmark_results_path: str | Path = "docs/benchmark_results.json",
    run_benchmark: bool = True,
) -> dict:
    """Returns {passed, promoted, results, failures}. `promoted` is only ever True when a
    model was actually loaded, run, and compared."""
    failures: list[str] = []
    results: dict[str, Any] = {}

    # --- 1. Serving-path regression gate -------------------------------------------
    if run_benchmark:
        completed = subprocess.run(
            [sys.executable, "scripts/benchmark.py", "--assert"], capture_output=True, text=True
        )
        if completed.returncode != 0:
            failures.append(f"Serving benchmark gate failed: {completed.stderr[-400:]}")

    benchmark_path = Path(benchmark_results_path)
    if benchmark_path.exists():
        results["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        failures.append("Benchmark results missing; run scripts/benchmark.py first.")

    # --- 2. Model-in-the-loop comparison -------------------------------------------
    if model_path is None:
        failures.append(
            "No adapter path supplied. The gate compares an adapter against its base model; "
            "there is nothing to promote."
        )
        return {"passed": False, "promoted": False, "results": results, "failures": failures}

    prompts = _held_out_prompts()
    base_answers = _generate(None, base_model, prompts)
    adapter_answers = _generate(str(model_path), base_model, prompts)

    if base_answers is None or adapter_answers is None:
        failures.append(
            "Could not run the model comparison (transformers/peft unavailable or model "
            "failed to load). Refusing to promote — this gate does not pass on absence of "
            "evidence. Install requirements-ml.txt to run it."
        )
        return {"passed": False, "promoted": False, "results": results, "failures": failures}

    base_score = _score_groundedness(base_answers, prompts)
    adapter_score = _score_groundedness(adapter_answers, prompts)
    gain = round(adapter_score - base_score, 4)
    results["groundedness"] = {
        "base_model": base_model,
        "base": base_score,
        "adapter": adapter_score,
        "gain": gain,
        "prompts": len(prompts),
        "min_gain_required": MIN_GROUNDEDNESS_GAIN,
        "min_absolute_required": MIN_ABSOLUTE_GROUNDEDNESS,
    }

    if adapter_score < MIN_ABSOLUTE_GROUNDEDNESS:
        failures.append(f"Adapter groundedness {adapter_score} < {MIN_ABSOLUTE_GROUNDEDNESS}")
    if gain < MIN_GROUNDEDNESS_GAIN:
        failures.append(
            f"Adapter improves groundedness by only {gain} over base "
            f"({base_score} -> {adapter_score}); minimum is {MIN_GROUNDEDNESS_GAIN}"
        )

    passed = not failures
    if passed:
        _write_active_model_pointer(model_path)
        logger.info("Eval gate passed: adapter promoted (%s -> %s).", base_score, adapter_score)
    else:
        logger.warning("Eval gate FAILED: %s", failures)

    return {"passed": passed, "promoted": passed, "results": results, "failures": failures}


def _write_active_model_pointer(model_path: str | Path) -> None:
    """Record the promoted adapter.

    Note for anyone reading this expecting the serving path to pick it up: it does not.
    The serving path is rule-based synthesis over retrieved evidence and loads no causal
    LM. This pointer records the outcome of the offline gate; wiring a generation backend
    into `SummaryService` is the work that would make it load-bearing, and until that
    exists nothing here should be described as "promoted to production".
    """
    pointer = Path("models/active_model.json")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "consumed_by_serving_path": False,
                "note": "Offline record only. SummaryService does not load a causal LM.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
