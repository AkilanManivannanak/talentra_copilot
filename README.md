<div align="center">

<img src="https://raw.githubusercontent.com/AkilanManivannanak/talentra_copilot/main/Talentra_copilot_cover.png" alt="Talentra Copilot" width="100%"/>

<br/>




![Talentra Copilot Banner](https://capsule-render.vercel.app/api?type=waving&color=0:2b0a3d,50:7c3aed,100:c084fc&height=220&section=header&text=Talentra%20Copilot🧠&fontSize=58&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Recruiting%20Intelligence%20-%20Zero%20External%20API%20Cost%20-%20Under%205ms%20p95%20Latency&descAlignY=60&descSize=20&animation=fadeIn)


<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

**[📺 Video Demo](https://drive.google.com/drive/folders/1Qr3iBMXZh5xUp5zWKCwhIW2AHBVrAtUm?usp=sharing)** &nbsp;|&nbsp;
**[🖼️ Screenshots](https://drive.google.com/drive/folders/1E5FJfOs-Gf_79UjTa6Q8U4JtIQ2FTyJF?usp=sharing)** &nbsp;|&nbsp;
**[🚀 Live Demo](https://drive.google.com/drive/folders/1Rx2Lzq3qjbHWWjdCU98pWKo9187kZzla?usp=sharing)** &nbsp;|&nbsp;
**[🐙 GitHub](https://github.com/AkilanManivannanak/talentra_copilot)**

</div>

---

## ⚡ TL;DR — What Was Built & Why It Matters

> **Talentra Copilot** is a full-stack AI recruiting assistant that ingests job descriptions and resumes, runs structured requirement-level evaluation across every candidate, and surfaces explainable rankings with cited evidence — all with **zero external API cost** and **sub-5ms p95 latency**.

```
✅  Evaluate p95:        4.81 ms     (SLO: <1500 ms)   → 99.7% headroom
✅  Copilot p95:         4.41 ms     (SLO: <1500 ms)   → 99.7% headroom  
✅  Upload p95:         18.76 ms     (partial-success fault-tolerant)
✅  Top-1 eval accuracy:  1.0        (hit@1 = 100%)
✅  Copilot consistency:  1.0        (ranking coherence = 100%)
✅  External API cost:   $0.000/req  (local-lexical by default)
✅  Error rate:           0.0%       (across 44 benchmarked requests)
```

---

## 🎯 Problem Statement & SLOs

| SLO | Target | Achieved |
|---|---|---|
| Evaluate endpoint p95 | ≤ 1,500 ms | **4.81 ms** |
| Copilot query p95 | ≤ 1,500 ms | **4.41 ms** |
| External cost per request | $0.00 | **$0.000** |
| Ranking correctness (top-1) | — | **1.0** |
| Copilot/Eval consistency | — | **1.0** |

The core SLO bets: (1) a recruiter must never wait more than 1.5 seconds for a ranking or answer, (2) the system must run for free at demo-scale with an upgrade path to semantic models, and (3) copilot answers must never contradict structured evaluation results.

---

## 🏗️ System Architecture

### Data Flow: Ingest → Store → Retrieve → Infer → Feedback

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIENT / UI LAYER                                │
│              Streamlit (Recruiter Dashboard)                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP / REST
┌───────────────────────────▼─────────────────────────────────────────┐
│                   FASTAPI SERVING LAYER                             │
│  ObservabilityMiddleware  │  CORS  │  /health  │  /ops/metrics      │
├──────────────┬────────────┴────────┬────────────┬───────────────────┤
│  /roles      │  /candidates        │  /evaluate │  /copilot/query   │
└──────┬───────┴──────────┬──────────┴─────┬──────┴──────────┬────────┘
       │                  │                │                 │
  ┌────▼────┐       ┌─────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
  │  Role   │       │ Document   │  │  Ranking    │  │  Copilot    │
  │ Ingest  │       │ Parser +   │  │  Service    │  │  Service    │
  │ & Req   │       │ Chunker    │  │  (cached)   │  │  (eval-first│
  │ Extract │       │            │  │             │  │   routing)  │
  └────┬────┘       └─────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                  │                │                │
  ┌────▼────────────────────────────────────────────────────▼──────┐
  │                    SERVICE CONTAINER                           │
  │   MetadataStore (JSON)  │  VectorStoreService (Lexical IDF)    │
  │   SummaryService        │  Evaluation Cache (dict, LRU-like)   │
  └────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │       DATA LAYER        │
              │  data/metadata.json     │
              │  data/vectorstore/      │
              │       index.json        │
              └─────────────────────────┘
```

### Request Lifecycle — Evaluate Flow

```
Recruiter clicks "Evaluate"
        │
        ▼
POST /roles/{role_id}/evaluate
        │
        ├── Check evaluation cache (cache_key = role_id + candidate_ids + store versions)
        │       └── HIT  → return deep copy immediately (sub-ms)
        │       └── MISS ↓
        │
        ├── For each candidate × requirement:
        │       └── VectorStore.search(query=requirement.text, k=2, filters={entity_id})
        │               └── Lexical IDF scoring + phrase bonus + contact-noise penalty
        │
        ├── Aggregate per-requirement scores → candidate total score
        │
        ├── Sort & rank candidates
        │
        └── Cache result → return EvaluateRoleResponse (with evidence citations)
```

### Request Lifecycle — Copilot Flow

```
Recruiter types question
        │
        ▼
POST /copilot/query
        │
        ├── Load evaluation results (cache hit expected)
        │
        ├── Classify question intent:
        │       ├── is_evaluation_question()  → route to eval results (comparison, ranking)
        │       └── extract_focus_query()     → route to targeted evidence search (skill deep-dive)
        │
        ├── Build answer with citations from evidence chunks
        │
        └── Return CopilotAnswerResponse {answer, citations, reasoning_trace}
```

---

## 🔍 Retrieval Design

Talentra uses a **local lexical retrieval engine** — purpose-built to be fast, transparent, and zero-cost at demo scale, with a clean upgrade path to dense/hybrid search.

| Layer | Implementation | Why |
|---|---|---|
| **Tokenisation** | Regex `[a-zA-Z0-9_+#.-]+` + stemming | Handles tech terms (C++, FastAPI, .NET) |
| **Scoring** | IDF-like term weighting | Down-weights ubiquitous terms |
| **Phrase bonus** | Bi-gram sequence matching | Rewards exact skill phrases |
| **Noise penalty** | Email/phone/URL pattern filter | Removes header junk from evidence |
| **Caching** | Dict-based with version invalidation | Zero-latency repeat queries |
| **Upgrade path** | `embedding_model` config flag | Swap to OpenAI/local dense with env var |

**Trade-off call:** Lexical retrieval is deterministic, debuggable, and quota-free. It sacrifices semantic recall (synonyms, paraphrases) for launch reliability. The architecture's abstraction layer means swapping to `text-embedding-3-small` is a single `.env` change.

---

## 📦 Project Structure

```
talentra_copilot/
├── app/
│   ├── main.py                   # FastAPI app + lifespan + middleware
│   ├── core/
│   │   ├── config.py             # Pydantic Settings (env-driven)
│   │   ├── logging.py            # Structured JSON logging
│   │   └── observability.py      # ObservabilityMiddleware + RequestMetricsStore
│   ├── routers/
│   │   ├── roles.py              # JD ingest + requirement extraction
│   │   ├── candidates.py         # Resume upload (partial-success fault tolerant)
│   │   ├── copilot.py            # /copilot/query endpoint
│   │   ├── ats.py                # ATS stage tracking + notes
│   │   ├── documents.py          # Raw document access
│   │   └── ops.py                # /ops/metrics, /ops/build-info
│   └── services/
│       ├── container.py          # Dependency injection container
│       ├── vectorstore.py        # Lexical retrieval engine (IDF + caching)
│       ├── ranking.py            # Requirement-level evaluation + result cache
│       ├── copilot.py            # Intent routing + answer synthesis
│       ├── requirement_extractor.py  # Rule-based JD parser
│       ├── document_parser.py    # PDF/DOCX chunker + noise filter
│       ├── summary.py            # Candidate profile summaries
│       ├── metadata_store.py     # JSON-backed metadata persistence
│       ├── candidates.py         # Candidate CRUD
│       ├── roles.py              # Role CRUD
│       ├── naming.py             # Entity deduplication helpers
│       └── ats.py                # ATS state machine
├── frontend/
│   └── streamlit_app.py          # Recruiter UI
├── eval/                         # Evaluation harness
├── scripts/                      # Utility scripts
├── tests/                        # Pytest suite
├── docs/
│   ├── architecture.md
│   ├── benchmark.md
│   ├── benchmark_results.json    # Raw benchmark data
│   └── postmortem.md
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

---

## 🚀 Quickstart

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/AkilanManivannanak/talentra_copilot.git
cd talentra_copilot

# (optional) copy and configure env
cp .env.example .env

docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Health check | http://localhost:8000/health |
| Metrics | http://localhost:8000/ops/metrics |

### Option 2 — Local (virtualenv)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Option 3 — Render (one-click cloud)

The `render.yaml` in the repo defines two services (`talentra-api` + `talentra-ui`). Connect the GitHub repo to [render.com](https://render.com) and deploy directly. No secrets required for the default local-lexical mode.

---

## ⚙️ Configuration

All config is environment-variable driven via Pydantic Settings:

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `local-rule-based` | `local-rule-based` or `gpt-4o-mini` |
| `EMBEDDING_MODEL` | `local-lexical` | `local-lexical` or `text-embedding-3-small` |
| `OPENAI_API_KEY` | `""` | Required only for OpenAI models |
| `DATA_DIR` | `./data` | Persistent data directory |
| `MATCH_THRESHOLD` | `0.28` | Minimum retrieval score |
| `MAX_UPLOAD_MB` | `8` | Per-file upload limit |
| `MAX_REQUIREMENTS` | `8` | Requirements extracted per JD |
| `LOG_JSON` | `false` | Structured JSON logs for prod |
| `METRICS_WINDOW_SIZE` | `1000` | Rolling window for ops metrics |
| `APP_ENV` | `local` | `local` / `render` |

---

## 📊 Benchmarks & Eval

All numbers are from `docs/benchmark_results.json` — in-process FastAPI TestClient on local CPU, no external API calls.

### Latency (p95)

| Endpoint | p95 | SLO | Status |
|---|---|---|---|
| `POST /roles/text` | 6.46 ms | ≤ 1,500 ms | ✅ |
| `POST /candidates/upload` | 18.76 ms | ≤ 1,500 ms | ✅ |
| `POST /roles/{id}/evaluate` | 4.81 ms | ≤ 1,500 ms | ✅ |
| `POST /copilot/query` | 4.41 ms | ≤ 1,500 ms | ✅ |
| Global p95 (all routes) | 3.55 ms | — | — |
| Global avg | 2.19 ms | — | — |

### Quality

| Metric | Value | Notes |
|---|---|---|
| Top-1 eval accuracy | **1.0** | 3 candidates, 8 requirements |
| Copilot ranking consistency | **1.0** | Eval ↔ Copilot coherence |
| Error rate | **0.0%** | 44 requests, 0 errors |

### Cost

| Mode | Cost/request | Notes |
|---|---|---|
| `local-lexical` (default) | **$0.000** | Zero external calls |
| `text-embedding-3-small` | ~$0.0001 | Upgrade path |
| `gpt-4o-mini` answers | ~$0.001–$0.004 | Upgrade path |

**How latency was reduced:** evaluation caching (cache_key = role_id + candidate_ids + store version) eliminates re-scoring on repeat queries. Lexical retrieval avoids network round-trips entirely. Both contribute to consistent sub-5ms p95.

---

## 🔧 Ops & Observability

Talentra ships with production observability built-in, not bolted on:

```
GET /health          → {"status": "ok"}
GET /ops/metrics     → rolling p50/p95/avg per route + global error rate
GET /ops/build-info  → git SHA, build timestamp, env
```

**ObservabilityMiddleware** records every request: route, latency, status code. The metrics store maintains a configurable rolling window (`METRICS_WINDOW_SIZE=1000`). Structured JSON logging (`LOG_JSON=true`) is ready for log aggregation in prod (Datadog, Loki, CloudWatch).

**Reliability design decisions:**
- Evaluation results are cached; cache is invalidated on store version bump (not TTL)
- Upload is partial-success: one bad PDF doesn't block the batch
- Copilot intent routing prevents eval/answer contradictions
- All endpoints return structured errors, never raw 500s

---

## 🩹 Postmortem: What Broke & How It Was Fixed

Five real incidents from development — each one made the system more resilient.

---

### Incident 1 — OpenAI Quota Failures Blocked All Ingestion

| | |
|---|---|
| **Symptom** | Resume upload returned HTTP 500; entire ingest pipeline dead |
| **Root cause** | Tight coupling to OpenAI embeddings; exhausted API quota = full outage |
| **Fix** | Default path replaced with local lexical retrieval. External API is opt-in via env var. Zero external dependency in the critical path. |
| **SRE lesson** | External API dependencies in the hot path are an SLO risk. Always have a local fallback. |

---

### Incident 2 — Brittle PDF Ingestion Caused Whole-Batch Failure

| | |
|---|---|
| **Symptom** | One malformed PDF crashed the entire upload request |
| **Root cause** | All-or-nothing processing; no per-file error isolation |
| **Fix** | Upload now returns partial success. Unreadable files return structured `{filename, error}` instead of bubbling a 500. |
| **SRE lesson** | Fan-out operations must fail independently. Bulkhead pattern. |

---

### Incident 3 — Ghost Candidates After Failed Ingest

| | |
|---|---|
| **Symptom** | Failed uploads left orphaned candidate records in metadata |
| **Root cause** | Metadata written before ingest fully succeeded (non-atomic) |
| **Fix** | Cleanup runs on failure path; orphan candidates are removed before response. |
| **SRE lesson** | Write-then-cleanup is not atomic. Prefer write-on-success or compensating transactions. |

---

### Incident 4 — Copilot Contradicted Evaluation Ranking

| | |
|---|---|
| **Symptom** | Evaluate ranked Akila #1; Copilot sometimes answered Jaxon or Esha for ranking questions |
| **Root cause** | Comparison questions routed to raw lexical hits instead of structured evaluation results |
| **Fix** | Intent classifier added. Ranking/comparison questions now consume evaluation output first. Skill deep-dives use evidence search separately. |
| **SRE lesson** | Two surfaces answering the same question from different data sources will diverge. Single source of truth for ranking. |

---

### Incident 5 — Header/Contact Junk Polluted Evidence

| | |
|---|---|
| **Symptom** | Citations surfaced email addresses, LinkedIn URLs, phone numbers |
| **Root cause** | Resume header chunks indexed without downweighting; high term frequency on contact tokens |
| **Fix** | Contact-noise penalties added to scoring; low-signal chunk filtering; evidence deduplication. |
| **SRE lesson** | Garbage in = garbage citations. Data quality gates belong at index time, not query time. |

---

## 🗺️ What's Next — Upgrade Path

The architecture is designed to scale without a rewrite:

| Capability | Current | Upgrade path |
|---|---|---|
| Retrieval | Local lexical IDF | Set `EMBEDDING_MODEL=text-embedding-3-small` |
| Answer generation | Rule-based templates | Set `CHAT_MODEL=gpt-4o-mini` |
| Storage | Local JSON | Swap to Postgres + pgvector |
| Search | Lexical only | Add OpenSearch for hybrid BM25 + dense |
| Reranking | None | Add cross-encoder reranker (ms-marco-MiniLM) |
| Eval gates | Manual | Add shadow test harness + auto rollback |
| CI/CD | Docker Compose | GitHub Actions → Render deploy hooks |

---

## 🧪 Tests & Eval Harness

```bash
# Run tests
pytest tests/ -v

# Run eval harness
python -m eval.run_eval

# View benchmark results
cat docs/benchmark_results.json | python -m json.tool
```

The eval harness measures top-k retrieval accuracy and copilot ranking consistency. Results are deterministic (no randomness in local-lexical mode), enabling regression detection on every push.

---

## 🏛️ Architecture Trade-offs Log

| Decision | Alternative | Why this |
|---|---|---|
| Local lexical retrieval | Dense embeddings (OpenAI) | Zero quota dependency; deterministic; debuggable. Dense is the upgrade path. |
| JSON file storage | Postgres / SQLite | Zero infra for demo. Clear migration target documented. |
| Rule-based extraction | LLM extraction | Demo-safe, no API key required, 100% reproducible. |
| Dict-based eval cache | Redis | Sub-process cache is sufficient at demo scale; Redis adds ops overhead. |
| Partial-success upload | Transactional batch | Partial success is strictly better UX for multi-file ingest. |
| Eval-first copilot routing | Single retrieval path | Prevents ranking contradictions across UI surfaces. |

---

## 👤 Author

**Akilan Manivannan**  
[GitHub](https://github.com/AkilanManivannanak) · [LinkedIn](https://linkedin.com/in/akilan-manivannan)

---

<div align="center">

**Built with zero external API cost. Designed for resilience. Optimized for recruiters.**

*Talentra Copilot — because hiring decisions deserve explainability.*

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2b0a3d,50:7c3aed,100:c084fc&height=120&section=footer" width="100%"/>
