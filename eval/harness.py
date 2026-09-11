"""
Retrieval evaluation harness.

Replaces the two metrics the old benchmark reported:

* `copilot_ranking_consistency` was `1.0 if top_candidate in copilot_answer`, but the
  answer is string-formatted *from* that candidate's name — it could not return anything
  but 1.0 for any input. Deleted.
* `top1_eval_accuracy` was `1.0 if top_candidate.lower().startswith('akila')` on a
  three-row fixture. Deleted.

What replaces them are standard ranking metrics computed against the graded judgements
in `eval/labels.json`, per requirement, averaged over requirements and roles:

    nDCG@k   — rewards putting strongly-relevant candidates above weakly-relevant ones,
               discounted by position. The headline number.
    P@k      — fraction of the top k that are relevant at all (relevance >= 2).
    MRR      — 1/rank of the first strongly-relevant candidate.
    Recall@k — fraction of all strongly-relevant candidates that made the top k.
    Kendall tau — rank correlation between the produced ordering and the ideal ordering.

These can fail. That is the point.

Run:
    python eval/harness.py                       # current configuration
    python eval/harness.py --ablation            # lexical vs dense vs hybrid vs +rerank
    python eval/harness.py --mode hybrid --k 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS_PATH = Path(__file__).resolve().parent / "labels.json"
RELEVANT_THRESHOLD = 2  # relevance >= 2 counts as "relevant" for binary metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))


def ndcg_at_k(ranked_relevances: Sequence[int], k: int) -> float:
    """Standard nDCG with the 2^rel - 1 gain function."""
    gains = [(2 ** rel) - 1 for rel in ranked_relevances[:k]]
    ideal = [(2 ** rel) - 1 for rel in sorted(ranked_relevances, reverse=True)[:k]]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def precision_at_k(ranked_relevances: Sequence[int], k: int) -> float:
    window = ranked_relevances[:k]
    return sum(1 for rel in window if rel >= RELEVANT_THRESHOLD) / len(window) if window else 0.0


def recall_at_k(ranked_relevances: Sequence[int], k: int) -> float:
    total_relevant = sum(1 for rel in ranked_relevances if rel >= RELEVANT_THRESHOLD)
    if not total_relevant:
        return 0.0
    return sum(1 for rel in ranked_relevances[:k] if rel >= RELEVANT_THRESHOLD) / total_relevant


def reciprocal_rank(ranked_relevances: Sequence[int]) -> float:
    for position, rel in enumerate(ranked_relevances, start=1):
        if rel >= RELEVANT_THRESHOLD:
            return 1.0 / position
    return 0.0


def kendall_tau(produced: Sequence[float], ideal: Sequence[float]) -> float:
    """Tau-b, computed directly so the harness has no scipy dependency."""
    n = len(produced)
    if n < 2:
        return 0.0
    concordant = discordant = tied_p = tied_i = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = produced[i] - produced[j]
            di = ideal[i] - ideal[j]
            if dp == 0 and di == 0:
                tied_p += 1
                tied_i += 1
            elif dp == 0:
                tied_p += 1
            elif di == 0:
                tied_i += 1
            elif (dp > 0) == (di > 0):
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    denominator = math.sqrt((total_pairs - tied_p) * (total_pairs - tied_i))
    return (concordant - discordant) / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_container(mode: str, rerank: bool, data_dir: str):
    os.environ["DATA_DIR"] = data_dir
    os.environ["VECTORSTORE_PATH"] = str(Path(data_dir) / "vs")
    os.environ["RETRIEVAL_MODE"] = mode
    os.environ["RERANK_ENABLED"] = "true" if rerank else "false"
    # Redaction removes candidate names, which is right for the product but would make
    # per-candidate filtering in the harness depend on names surviving. Filtering is by id.
    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.services.container import ServiceContainer

    return ServiceContainer.from_settings(get_settings())


def evaluate_config(
    labels: dict,
    *,
    mode: str = "lexical",
    rerank: bool = False,
    k: int = 5,
) -> dict[str, Any]:
    """Index every candidate once, then score each requirement as a query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        services = _build_container(mode, rerank, tmpdir)
        retriever = services.retriever

        started = time.perf_counter()
        candidate_index: dict[str, str] = {}
        for candidate in labels["candidates"]:
            retriever.add_document(
                text=candidate["text"],
                metadata_base={
                    "document_id": candidate["id"],
                    "entity_type": "candidate",
                    "entity_id": candidate["id"],
                    "entity_name": candidate["name"],
                    "filename": candidate["filename"],
                },
            )
            candidate_index[candidate["id"]] = candidate["name"]
        index_seconds = time.perf_counter() - started

        judgement_lookup = {
            (j["role_id"], j["requirement_id"], j["candidate_id"]): j["relevance"]
            for j in labels["judgements"]
        }

        per_query: list[dict[str, Any]] = []
        query_latencies: list[float] = []

        for role in labels["roles"]:
            for requirement in role["requirements"]:
                pool = role["candidate_ids"]
                # One corpus-wide retrieval, grouped by candidate — the same call the
                # RankingService makes, so the harness measures the shipped path.
                started = time.perf_counter()
                grouped = retriever.search_grouped(
                    query=requirement["text"], per_entity_k=2, filters={"entity_type": "candidate"}
                )
                query_latencies.append((time.perf_counter() - started) * 1000)
                scores = [
                    (cid, max((hit["score"] for hit in grouped.get(cid, [])), default=0.0))
                    for cid in pool
                ]

                scores.sort(key=lambda item: item[1], reverse=True)
                ranked_relevances = [
                    judgement_lookup.get((role["id"], requirement["id"], cid), 0) for cid, _ in scores
                ]
                produced_scores = [score for _, score in scores]
                ideal_scores = [float(rel) for rel in ranked_relevances]

                per_query.append(
                    {
                        "role_id": role["id"],
                        "requirement_id": requirement["id"],
                        "requirement": requirement["text"],
                        f"ndcg@{k}": round(ndcg_at_k(ranked_relevances, k), 4),
                        f"precision@{min(3, k)}": round(precision_at_k(ranked_relevances, min(3, k)), 4),
                        f"recall@{k}": round(recall_at_k(ranked_relevances, k), 4),
                        "mrr": round(reciprocal_rank(ranked_relevances), 4),
                        "kendall_tau": round(kendall_tau(produced_scores, ideal_scores), 4),
                        "top_3": [candidate_index[cid] for cid, _ in scores[:3]],
                    }
                )

        def average(metric: str) -> float:
            return round(mean(item[metric] for item in per_query), 4)

        latencies = sorted(query_latencies)
        p95_index = max(0, min(len(latencies) - 1, round(0.95 * (len(latencies) - 1))))

        return {
            "config": {
                "requested_mode": mode,
                "effective_mode": retriever.mode,
                "rerank_requested": rerank,
                "rerank_active": retriever.rerank_active,
                "embedding_model": services.settings.embedding_model if retriever.mode != "lexical" else None,
                "k": k,
            },
            "corpus": {
                "candidates_indexed": len(labels["candidates"]),
                "queries": len(per_query),
                "judgements": len(labels["judgements"]),
                "index_seconds": round(index_seconds, 3),
            },
            "metrics": {
                f"ndcg@{k}": average(f"ndcg@{k}"),
                f"precision@{min(3, k)}": average(f"precision@{min(3, k)}"),
                f"recall@{k}": average(f"recall@{k}"),
                "mrr": average("mrr"),
                "kendall_tau": average("kendall_tau"),
            },
            "retrieval_latency_ms": {
                "mean": round(mean(query_latencies), 3) if query_latencies else 0.0,
                "p95": round(latencies[p95_index], 3) if latencies else 0.0,
                "samples": len(latencies),
            },
            "per_query": per_query,
        }


def worst_queries(result: dict, k: int, limit: int = 5) -> list[dict]:
    """The queries the system is worst at. Reported deliberately — an eval you only
    read the average of is a scoreboard, not a diagnostic."""
    metric = f"ndcg@{k}"
    ordered = sorted(result["per_query"], key=lambda item: item[metric])
    return [
        {"requirement": item["requirement"][:90], metric: item[metric], "top_3": item["top_3"]}
        for item in ordered[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Talentra retrieval evaluation")
    parser.add_argument("--mode", default="lexical", choices=["lexical", "dense", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ablation", action="store_true", help="run every configuration and print a table")
    parser.add_argument("--out", default="docs/eval_results.json")
    args = parser.parse_args()

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    print(f"Eval set: {labels['counts']} \n{labels['method']}\n")

    configs = (
        [("lexical", False), ("dense", False), ("hybrid", False), ("hybrid", True)]
        if args.ablation
        else [(args.mode, args.rerank)]
    )

    results = []
    for mode, rerank in configs:
        print(f"running: mode={mode} rerank={rerank} ...", flush=True)
        results.append(evaluate_config(labels, mode=mode, rerank=rerank, k=args.k))

    ndcg_label = f"nDCG@{args.k}"
    recall_label = f"Recall@{args.k}"
    header = (f"{'configuration':<24}{ndcg_label:>10}{'P@3':>9}"
              f"{recall_label:>11}{'MRR':>9}{'tau':>9}{'p95 ms':>10}")
    print("\n" + header)
    print("-" * len(header))
    for result in results:
        config = result["config"]
        label = config["effective_mode"] + (" + rerank" if config["rerank_active"] else "")
        if config["requested_mode"] != config["effective_mode"]:
            label += f" (asked {config['requested_mode']})"
        metrics = result["metrics"]
        print(
            f"{label:<24}"
            f"{metrics[f'ndcg@{args.k}']:>10.4f}"
            f"{metrics['precision@3']:>9.4f}"
            f"{metrics[f'recall@{args.k}']:>11.4f}"
            f"{metrics['mrr']:>9.4f}"
            f"{metrics['kendall_tau']:>9.4f}"
            f"{result['retrieval_latency_ms']['p95']:>10.3f}"
        )

    print(f"\nWorst {args.k} queries for the last configuration:")
    for item in worst_queries(results[-1], args.k):
        print(f"  {item[f'ndcg@{args.k}']:.3f}  {item['requirement']}")
        print(f"         top-3: {', '.join(item['top_3'])}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval_set": {"method": labels["method"], "limitation": labels["limitation"], **labels["counts"]},
        "configurations": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
