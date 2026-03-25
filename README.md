# Talentra Copilot

**Talentra Copilot** is a local-first, evidence-grounded hiring intelligence system that combines:
- resume and JD ingestion
- requirement-aware candidate ranking
- evaluation-aware recruiter Q&A
- ATS-lite workflow management
- request metrics, CI, Docker, and deployment scaffolding

## Why this version is materially stronger

This is no longer just a “resume screener” demo. It now shows the engineering signals hiring managers actually look for:
- **Architecture and trade-offs** documented from ingest → retrieval → serving → feedback
- **Measured metrics** with a reproducible benchmark script
- **Observability** via request IDs and route-level latency summaries
- **Reliability** via partial-success batch upload, parser hardening, and cleanup of failed ingests
- **Deployment evidence** via Docker, docker-compose, Render config, and GitHub Actions CI
- **Postmortem discipline** documenting what broke and how the system was fixed

## System goal and SLOs

Primary goal: help recruiters screen candidates with grounded evidence while keeping the system cheap, stable, and locally runnable.

Target SLOs:
- **Evaluate p95 latency** < 1.5 s
- **Copilot Q&A p95 latency** < 1.5 s
- **External model/API cost per request** = $0.00 by default
- **Copilot ranking consistency** = 1.0 on the included benchmark fixture

## Architecture sketch

```text
UI (Streamlit)
   |
   v
FastAPI serving layer
   |
   +--> role ingestion -> requirement extraction -> metadata store
   +--> candidate ingestion -> parser -> chunker -> lexical index
   +--> evaluation service -> retrieve evidence per requirement -> aggregate scores
   +--> copilot router -> evaluation-aware answers OR targeted evidence search
   +--> ATS workflow -> stage transitions + recruiter notes
   +--> ops endpoints -> request metrics + build metadata
```

More detail: [docs/architecture.md](docs/architecture.md)

## Metrics snapshot

<!-- BENCHMARK_METRICS_START -->
- **Role create p95**: 6.46 ms
- **Candidate upload p95**: 18.76 ms
- **Evaluate p95**: 4.81 ms
- **Copilot p95**: 4.41 ms
- **Top-1 evaluation accuracy** (included fixture): 1.0
- **Copilot ranking consistency** (included fixture): 1.0
- **External API cost / request**: $0.000
<!-- BENCHMARK_METRICS_END -->

## Data flow

1. **Ingest**: parse JD/resume text locally from PDF, DOCX, TXT, or Markdown.
2. **Store**: persist metadata in `data/metadata.json` and indexed chunks in `data/vectorstore/index.json`.
3. **Retrieve**: lexical search scores chunks with IDF-style weighting, phrase bonuses, and noise penalties.
4. **Infer**: requirement-level evidence is aggregated into candidate scores and grounded summaries.
5. **Serve**: FastAPI exposes APIs; Streamlit provides role creation, candidate upload, ranking, ATS actions, and Copilot Q&A.
6. **Feedback**: recruiter notes and stage changes are stored and surfaced in the ATS dashboard.

## Reliability and trade-offs

- **Latency vs quality**: lexical retrieval is much faster and cheaper than dense retrieval + rerankers, but weaker semantically.
- **Freshness vs cost**: local indexing avoids quota failures and API spend, but gives up cloud-model semantics.
- **Resilience**: uploads support partial success; failed ingests are cleaned up; Q&A routes comparison questions through evaluation results first.
- **Observability**: every request gets an `X-Request-ID`; `/ops/metrics` exposes latency summaries by route.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

In a second terminal:

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

## Run with Docker

```bash
docker compose up --build
```

API: `http://localhost:8000`
UI: `http://localhost:8501`

## CI / CD and ops

- **CI**: `.github/workflows/ci.yml` runs tests, coverage, and a benchmark gate.
- **Docker**: `Dockerfile` and `docker-compose.yml` support containerized local runs.
- **Cloud deploy stub**: `render.yaml` shows how to deploy API and UI as separate web services.
- **Ops endpoints**:
  - `GET /health`
  - `GET /ops/metrics`
  - `GET /ops/build`

## Benchmarking

```bash
python scripts/benchmark.py
```

Outputs:
- `docs/benchmark_results.json`
- `docs/benchmark.md`

## Demo flow

1. Create a role from JD text.
2. Upload 2–4 resumes.
3. Run evaluation.
4. Use ATS Workflow to shortlist and add recruiter notes.
5. Ask Copilot questions such as:
   - `Based on the evaluation results, who is the strongest candidate for this role?`
   - `Why is Akila ranked above Jaxon?`
   - `Compare the selected candidates against the role requirements and explain the ranking.`
   - `Who shows the strongest evidence for machine learning?`

## Postmortem

Read: [docs/postmortem.md](docs/postmortem.md)

## What this still is not

This is **not** pretending to be a production semantic stack with dense embeddings, rerankers, feature stores, or online learning. It is a local-first AI engineering portfolio project that now demonstrates the missing engineering discipline: architecture, metrics, CI, deployability, observability, and postmortem thinking.
