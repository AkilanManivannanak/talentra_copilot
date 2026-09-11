"""
Build the Talentra evaluation set.

Method, stated plainly because a metric whose labelling process is invisible is not a
metric:

* The resumes are **synthetic**. Using real resumes would mean processing other people's
  personal data to publish a benchmark, which is not a trade this project is willing to
  make. Every candidate is authored here.
* Each candidate is defined by a structured **evidence profile**: a mapping from
  capability -> {strong, partial, none}, plus a seniority band. The prose is then written
  from that profile using *paraphrases* — the resume says "shipped a semantic search
  service over 40M documents", not "RAG: strong" — so the surface form does not hand the
  answer to a lexical matcher.
* The **relevance judgements are derived from the profile, not from the prose**, on the
  standard graded scale (3 = strong direct evidence, 2 = clear evidence, 1 = weak or
  adjacent, 0 = none). The retrieval system never sees the profile.

Known limitation, which the README repeats: synthetic prose is cleaner and more
consistent than real resumes, so absolute scores here are optimistic. The set is built
for *relative* comparison between retrieval configurations, which is what the ablation
in `eval/harness.py` uses it for.

Run:  python eval/build_labels.py            # writes eval/labels.json
"""
from __future__ import annotations

import json
from pathlib import Path

STRONG, CLEAR, WEAK, NONE = 3, 2, 1, 0

# --------------------------------------------------------------------------------------
# Capabilities. Each maps to prose fragments at three evidence levels.
# Fragments deliberately avoid the capability's own name where a practitioner would.
# --------------------------------------------------------------------------------------
CAPABILITIES: dict[str, dict[int, list[str]]] = {
    "python_depth": {
        STRONG: [
            "Six years writing production services in Python; owned the typing migration across a 200k-line codebase.",
            "Primary language for five years: async services, packaging, and CPython profiling for hot paths.",
        ],
        CLEAR: [
            "Three years of day-to-day Python across data services and internal tooling.",
            "Built and maintained several Python microservices over the last two years.",
        ],
        WEAK: [
            "Used Python for coursework, notebooks, and a capstone project.",
            "Comfortable scripting in Python; most professional work was in Java.",
        ],
        NONE: ["Primarily a frontend engineer working in TypeScript and Swift."],
    },
    "rag_systems": {
        STRONG: [
            "Designed and shipped a retrieval-augmented assistant over 40M internal documents, including chunking strategy, "
            "hybrid search, and a citation layer that grounded every generated claim.",
            "Owned a production question-answering system backed by semantic search; ran the chunking and reranking experiments "
            "that took answer groundedness from 0.61 to 0.88.",
        ],
        CLEAR: [
            "Built a document question-answering prototype using vector search over embedded passages, deployed internally.",
            "Implemented semantic search over a knowledge base and wired it into an LLM answer endpoint.",
        ],
        WEAK: [
            "Followed a tutorial to build a chatbot over PDFs for a hackathon weekend.",
            "Read widely on retrieval augmentation; no production implementation yet.",
        ],
        NONE: ["Work has been in classical tabular modelling; no retrieval or LLM systems."],
    },
    "eval_rigour": {
        STRONG: [
            "Built the offline evaluation harness the search team ranks releases on: graded relevance judgements, nDCG and MRR "
            "dashboards, and a regression gate wired into CI.",
            "Established the labelling guidelines and inter-annotator agreement process for our ranking quality programme.",
        ],
        CLEAR: [
            "Ran A/B tests and offline precision/recall analysis before each model rollout.",
            "Wrote the metric pipeline that tracked model quality release over release.",
        ],
        WEAK: [
            "Reported accuracy and F1 on a held-out split for course projects.",
            "Checked model outputs manually before shipping.",
        ],
        NONE: ["No formal evaluation practice; correctness was assessed by code review."],
    },
    "serving_infra": {
        STRONG: [
            "Own the serving stack: FastAPI behind a load balancer, containerised, autoscaled on EKS, with SLO dashboards and paging.",
            "Ran the platform's request-serving layer — containers, blue-green rollout, and latency budgets per endpoint.",
        ],
        CLEAR: [
            "Built REST services with FastAPI and shipped them as Docker images to a managed cloud runtime.",
            "Containerised our model services and deployed them to Google Cloud Run.",
        ],
        WEAK: [
            "Deployed a Streamlit demo to a free hosting tier.",
            "Wrote a Dockerfile once for a class assignment.",
        ],
        NONE: ["All work has been in notebooks handed to an engineering team to productionise."],
    },
    "agent_orchestration": {
        STRONG: [
            "Built a multi-step agent workflow with conditional routing, checkpointed state, and a mandatory human approval step "
            "before any write to the system of record.",
            "Designed the orchestration layer for our tool-using assistant: typed state, retries, and an approval gate on side effects.",
        ],
        CLEAR: [
            "Chained several LLM calls with tool use and branching logic for an internal automation.",
            "Used a graph-based orchestration framework to sequence model calls with fallbacks.",
        ],
        WEAK: [
            "Experimented with agent frameworks on a side project.",
            "Familiar with function calling from reading the API docs.",
        ],
        NONE: ["No experience with multi-step or tool-using model workflows."],
    },
    "data_pipelines": {
        STRONG: [
            "Own the batch and streaming pipelines feeding the feature store — Spark, Airflow, and schema contracts across teams.",
            "Built the ingestion layer processing 3TB/day with backfill and late-arrival handling.",
        ],
        CLEAR: [
            "Wrote scheduled ETL jobs in Airflow feeding the analytics warehouse.",
            "Maintained dbt models and the SQL transformations behind reporting.",
        ],
        WEAK: [
            "Wrote ad-hoc SQL queries and pandas scripts for analysis.",
            "Cleaned datasets by hand for research projects.",
        ],
        NONE: ["No data engineering exposure."],
    },
    "ml_modelling": {
        STRONG: [
            "Trained and shipped ranking and classification models end to end, including feature design, calibration, and drift monitoring.",
            "Led model development for the recommendation surface; owned training, offline eval, and online rollout.",
        ],
        CLEAR: [
            "Built supervised models with scikit-learn and PyTorch for internal prediction tasks.",
            "Fine-tuned transformer classifiers for document routing.",
        ],
        WEAK: [
            "Completed several machine learning courses and Kaggle competitions.",
            "Trained baseline models for a university project.",
        ],
        NONE: ["Background is pure backend engineering; no modelling work."],
    },
}

SENIORITY_PROSE = {
    "senior": "Senior engineer, {years} years professional experience.",
    "mid": "Engineer with {years} years of professional experience.",
    "junior": "Early-career engineer, {years} year(s) of professional experience.",
    "student": "Graduate student; internships and coursework, no full-time industry role yet.",
}

# --------------------------------------------------------------------------------------
# Candidate pool. Each entry: id, display name, seniority, years, evidence profile.
# Profiles are varied on purpose: several candidates are strong on one axis and empty on
# another, which is what separates a good ranker from one that just scores verbosity.
# --------------------------------------------------------------------------------------
P = dict  # readability

CANDIDATE_POOL = [
    ("c01", "Priya Raman", "senior", 7, P(python_depth=STRONG, rag_systems=STRONG, eval_rigour=STRONG, serving_infra=STRONG, agent_orchestration=CLEAR, data_pipelines=CLEAR, ml_modelling=STRONG)),
    ("c02", "Daniel Okafor", "senior", 8, P(python_depth=STRONG, rag_systems=CLEAR, eval_rigour=CLEAR, serving_infra=STRONG, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=CLEAR)),
    ("c03", "Mei Lin Chow", "mid", 4, P(python_depth=STRONG, rag_systems=STRONG, eval_rigour=CLEAR, serving_infra=CLEAR, agent_orchestration=STRONG, data_pipelines=WEAK, ml_modelling=CLEAR)),
    ("c04", "Tomas Vidal", "mid", 3, P(python_depth=CLEAR, rag_systems=CLEAR, eval_rigour=WEAK, serving_infra=CLEAR, agent_orchestration=CLEAR, data_pipelines=CLEAR, ml_modelling=WEAK)),
    ("c05", "Ayesha Siddiqui", "senior", 9, P(python_depth=CLEAR, rag_systems=NONE, eval_rigour=STRONG, serving_infra=STRONG, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=STRONG)),
    ("c06", "Jonas Beck", "junior", 1, P(python_depth=CLEAR, rag_systems=WEAK, eval_rigour=WEAK, serving_infra=WEAK, agent_orchestration=WEAK, data_pipelines=WEAK, ml_modelling=CLEAR)),
    ("c07", "Fatima Zahra", "mid", 5, P(python_depth=STRONG, rag_systems=CLEAR, eval_rigour=STRONG, serving_infra=CLEAR, agent_orchestration=WEAK, data_pipelines=CLEAR, ml_modelling=STRONG)),
    ("c08", "Ruben Alvarez", "student", 0, P(python_depth=WEAK, rag_systems=WEAK, eval_rigour=WEAK, serving_infra=NONE, agent_orchestration=NONE, data_pipelines=WEAK, ml_modelling=WEAK)),
    ("c09", "Hana Kobayashi", "senior", 6, P(python_depth=STRONG, rag_systems=STRONG, eval_rigour=CLEAR, serving_infra=CLEAR, agent_orchestration=STRONG, data_pipelines=NONE, ml_modelling=CLEAR)),
    ("c10", "Samuel Adeyemi", "mid", 4, P(python_depth=CLEAR, rag_systems=NONE, eval_rigour=CLEAR, serving_infra=STRONG, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=WEAK)),
    ("c11", "Elena Petrova", "senior", 10, P(python_depth=STRONG, rag_systems=WEAK, eval_rigour=STRONG, serving_infra=CLEAR, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=STRONG)),
    ("c12", "Kwame Mensah", "junior", 2, P(python_depth=CLEAR, rag_systems=CLEAR, eval_rigour=WEAK, serving_infra=WEAK, agent_orchestration=CLEAR, data_pipelines=WEAK, ml_modelling=CLEAR)),
    ("c13", "Sofia Marchetti", "mid", 5, P(python_depth=CLEAR, rag_systems=STRONG, eval_rigour=CLEAR, serving_infra=CLEAR, agent_orchestration=CLEAR, data_pipelines=CLEAR, ml_modelling=CLEAR)),
    ("c14", "Arjun Nair", "student", 0, P(python_depth=WEAK, rag_systems=CLEAR, eval_rigour=WEAK, serving_infra=WEAK, agent_orchestration=WEAK, data_pipelines=NONE, ml_modelling=CLEAR)),
    ("c15", "Nora Lindqvist", "senior", 7, P(python_depth=NONE, rag_systems=NONE, eval_rigour=CLEAR, serving_infra=STRONG, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=NONE)),
    ("c16", "Ibrahim Toure", "mid", 3, P(python_depth=STRONG, rag_systems=WEAK, eval_rigour=CLEAR, serving_infra=CLEAR, agent_orchestration=WEAK, data_pipelines=CLEAR, ml_modelling=STRONG)),
    ("c17", "Wei Zhang", "senior", 8, P(python_depth=STRONG, rag_systems=CLEAR, eval_rigour=STRONG, serving_infra=STRONG, agent_orchestration=CLEAR, data_pipelines=CLEAR, ml_modelling=STRONG)),
    ("c18", "Lucia Fernandes", "junior", 2, P(python_depth=CLEAR, rag_systems=WEAK, eval_rigour=NONE, serving_infra=CLEAR, agent_orchestration=NONE, data_pipelines=CLEAR, ml_modelling=WEAK)),
    ("c19", "Omar Haddad", "mid", 4, P(python_depth=CLEAR, rag_systems=STRONG, eval_rigour=WEAK, serving_infra=WEAK, agent_orchestration=STRONG, data_pipelines=NONE, ml_modelling=CLEAR)),
    ("c20", "Grace Mwangi", "senior", 6, P(python_depth=STRONG, rag_systems=NONE, eval_rigour=STRONG, serving_infra=CLEAR, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=CLEAR)),
    ("c21", "Viktor Novak", "mid", 5, P(python_depth=CLEAR, rag_systems=CLEAR, eval_rigour=CLEAR, serving_infra=STRONG, agent_orchestration=CLEAR, data_pipelines=WEAK, ml_modelling=WEAK)),
    ("c22", "Aisha Rahman", "student", 0, P(python_depth=CLEAR, rag_systems=WEAK, eval_rigour=WEAK, serving_infra=NONE, agent_orchestration=WEAK, data_pipelines=WEAK, ml_modelling=CLEAR)),
    ("c23", "Marcus Bell", "senior", 11, P(python_depth=CLEAR, rag_systems=NONE, eval_rigour=CLEAR, serving_infra=STRONG, agent_orchestration=NONE, data_pipelines=STRONG, ml_modelling=NONE)),
    ("c24", "Yuki Tanaka", "mid", 4, P(python_depth=STRONG, rag_systems=STRONG, eval_rigour=STRONG, serving_infra=WEAK, agent_orchestration=CLEAR, data_pipelines=WEAK, ml_modelling=STRONG)),
    ("c25", "Diego Ramirez", "junior", 1, P(python_depth=WEAK, rag_systems=NONE, eval_rigour=NONE, serving_infra=WEAK, agent_orchestration=NONE, data_pipelines=CLEAR, ml_modelling=WEAK)),
    ("c26", "Leila Haddadi", "mid", 6, P(python_depth=STRONG, rag_systems=CLEAR, eval_rigour=CLEAR, serving_infra=CLEAR, agent_orchestration=STRONG, data_pipelines=CLEAR, ml_modelling=CLEAR)),
    ("c27", "Peter Svensson", "senior", 9, P(python_depth=CLEAR, rag_systems=WEAK, eval_rigour=STRONG, serving_infra=CLEAR, agent_orchestration=WEAK, data_pipelines=STRONG, ml_modelling=STRONG)),
    ("c28", "Ananya Iyer", "mid", 3, P(python_depth=CLEAR, rag_systems=CLEAR, eval_rigour=CLEAR, serving_infra=CLEAR, agent_orchestration=CLEAR, data_pipelines=CLEAR, ml_modelling=CLEAR)),
    ("c29", "Thomas Okonkwo", "student", 0, P(python_depth=WEAK, rag_systems=WEAK, eval_rigour=WEAK, serving_infra=WEAK, agent_orchestration=CLEAR, data_pipelines=NONE, ml_modelling=WEAK)),
    ("c30", "Camille Dubois", "senior", 7, P(python_depth=STRONG, rag_systems=STRONG, eval_rigour=CLEAR, serving_infra=STRONG, agent_orchestration=CLEAR, data_pipelines=CLEAR, ml_modelling=CLEAR)),
]

# --------------------------------------------------------------------------------------
# Roles. Each requirement names the capability it tests, so judgements are derivable.
# --------------------------------------------------------------------------------------
ROLES = [
    {
        "id": "r_ai_engineer",
        "title": "AI Engineer, Search & Retrieval",
        "description": """AI Engineer, Search & Retrieval

Requirements:
- Must have 3+ years of professional Python experience.
- Required: experience designing and shipping retrieval-augmented generation systems in production.
- Strong knowledge of building and deploying HTTP services with FastAPI and Docker on a cloud platform.
- Experience designing offline evaluation pipelines and measuring retrieval quality with ranking metrics.
Nice to have: exposure to agent orchestration frameworks and multi-step LLM workflows.
""",
        "requirements": [
            ("req_py", "Must have 3+ years of professional Python experience", "python_depth", True),
            ("req_rag", "Required: experience designing and shipping retrieval-augmented generation systems in production", "rag_systems", True),
            ("req_serve", "Strong knowledge of building and deploying HTTP services with FastAPI and Docker on a cloud platform", "serving_infra", False),
            ("req_eval", "Experience designing offline evaluation pipelines and measuring retrieval quality with ranking metrics", "eval_rigour", False),
            ("req_agent", "Nice to have: exposure to agent orchestration frameworks and multi-step LLM workflows", "agent_orchestration", False),
        ],
        "candidates": ["c01", "c03", "c05", "c07", "c08", "c09", "c13", "c15", "c19", "c24", "c26", "c30"],
    },
    {
        "id": "r_ml_platform",
        "title": "Machine Learning Platform Engineer",
        "description": """Machine Learning Platform Engineer

Requirements:
- Must have strong Python engineering skills in a production codebase.
- Required: experience operating batch or streaming data pipelines at scale.
- Experience owning model serving infrastructure, containers, and deployment.
- Demonstrated ability to train and ship machine learning models end to end.
Nice to have: experience with retrieval systems or vector search.
""",
        "requirements": [
            ("req_py", "Must have strong Python engineering skills in a production codebase", "python_depth", True),
            ("req_pipe", "Required: experience operating batch or streaming data pipelines at scale", "data_pipelines", True),
            ("req_serve", "Experience owning model serving infrastructure, containers, and deployment", "serving_infra", False),
            ("req_model", "Demonstrated ability to train and ship machine learning models end to end", "ml_modelling", False),
            ("req_rag", "Nice to have: experience with retrieval systems or vector search", "rag_systems", False),
        ],
        "candidates": ["c02", "c05", "c10", "c11", "c15", "c16", "c17", "c20", "c23", "c25", "c27", "c28"],
    },
    {
        "id": "r_applied_scientist",
        "title": "Applied Scientist, Agentic Systems",
        "description": """Applied Scientist, Agentic Systems

Requirements:
- Must have experience building multi-step agent or tool-using LLM workflows.
- Required: rigorous offline evaluation practice with graded relevance or preference data.
- Strong Python skills for research and production code.
- Experience with retrieval-augmented systems and grounding generated output in evidence.
Nice to have: experience training or fine-tuning models.
""",
        "requirements": [
            ("req_agent", "Must have experience building multi-step agent or tool-using LLM workflows", "agent_orchestration", True),
            ("req_eval", "Required: rigorous offline evaluation practice with graded relevance or preference data", "eval_rigour", True),
            ("req_py", "Strong Python skills for research and production code", "python_depth", False),
            ("req_rag", "Experience with retrieval-augmented systems and grounding generated output in evidence", "rag_systems", False),
            ("req_model", "Nice to have: experience training or fine-tuning models", "ml_modelling", False),
        ],
        "candidates": ["c01", "c03", "c04", "c06", "c09", "c12", "c14", "c19", "c21", "c24", "c26", "c29"],
    },
]


def build_resume(index: int, name: str, seniority: str, years: int, profile: dict) -> str:
    """Write prose from the profile, choosing a different paraphrase per candidate.

    The alternating fragment choice matters: if every 'STRONG rag_systems' candidate got
    identical wording, lexical retrieval would score artificially well and the hybrid
    ablation would be meaningless.
    """
    lines = [name, SENIORITY_PROSE[seniority].format(years=years), ""]
    for capability, level in profile.items():
        options = CAPABILITIES[capability][level]
        lines.append("- " + options[index % len(options)])
    return "\n".join(lines)


def main() -> None:
    pool = {cid: (name, seniority, years, profile) for cid, name, seniority, years, profile in CANDIDATE_POOL}

    candidates = []
    for index, (cid, name, seniority, years, profile) in enumerate(CANDIDATE_POOL):
        candidates.append(
            {
                "id": cid,
                "name": name,
                "filename": f"{name.lower().replace(' ', '_')}_resume.txt",
                "text": build_resume(index, name, seniority, years, profile),
                # Retained for auditability of the labels; never shown to the retriever.
                "_profile": profile,
                "_seniority": seniority,
            }
        )

    roles, judgements = [], []
    for role in ROLES:
        roles.append(
            {
                "id": role["id"],
                "title": role["title"],
                "description": role["description"],
                "requirements": [
                    {"id": rid, "text": text, "capability": cap, "must_have": must}
                    for rid, text, cap, must in role["requirements"]
                ],
                "candidate_ids": role["candidates"],
            }
        )
        for rid, _text, capability, _must in role["requirements"]:
            for cid in role["candidates"]:
                _name, _sen, _yrs, profile = pool[cid]
                judgements.append(
                    {
                        "role_id": role["id"],
                        "requirement_id": rid,
                        "candidate_id": cid,
                        "relevance": profile[capability],
                    }
                )

    payload = {
        "schema_version": 2,
        "method": (
            "Synthetic resumes authored from structured evidence profiles; graded relevance "
            "(0-3) derived from the profile, not from the prose. See eval/build_labels.py."
        ),
        "grading_scale": {"3": "strong direct evidence", "2": "clear evidence", "1": "weak or adjacent", "0": "none"},
        "limitation": (
            "Synthetic prose is cleaner and more consistent than real resumes, so absolute "
            "scores are optimistic. Use this set for relative comparison between retrieval "
            "configurations, not as an estimate of real-world accuracy."
        ),
        "counts": {
            "roles": len(roles),
            "candidates": len(candidates),
            "requirements": sum(len(r["requirements"]) for r in roles),
            "judgements": len(judgements),
        },
        "roles": roles,
        "candidates": candidates,
        "judgements": judgements,
    }

    out = Path(__file__).resolve().parent / "labels.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
