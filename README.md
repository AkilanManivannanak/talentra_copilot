<div align="center">
![Talentra Copilot Banner](https://capsule-render.vercel.app/api?type=waving&color=0:2b0a3d,50:7c3aed,100:c084fc&height=220&section=header&text=Talentra%20Copilot🧠&fontSize=58&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Recruiting%20Intelligence%20-%20Zero%20External%20API%20Cost%20-%20Under%205ms%20p95%20Latency&descAlignY=60&descSize=20&animation=fadeIn)


<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-FF6B35?style=for-the-badge&logo=graphql&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![LoRA](https://img.shields.io/badge/LoRA-Fine--tuning-8A2BE2?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/peft)
[![DPO](https://img.shields.io/badge/DPO-Alignment-C084FC?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/trl/dpo_trainer)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/AkilanManivannanak/talentra_copilot/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

**[🎬 Video Demo](https://drive.google.com/drive/folders/1Rx2Lzq3qjbHWWjdCU98pWKo9187kZzla?usp=sharing)** &nbsp;|&nbsp;
**[📸 Screenshots](https://drive.google.com/drive/folders/1Rx2Lzq3qjbHWWjdCU98pWKo9187kZzla?usp=sharing)** &nbsp;|&nbsp;
**[🐙 GitHub](https://github.com/AkilanManivannanak/talentra_copilot)** &nbsp;|&nbsp;
**[📖 Architecture](docs/architecture.md)** &nbsp;|&nbsp;
**[📊 Benchmark](docs/benchmark_results.json)**

<br/>

> *Compound AI hiring intelligence — multi-agent orchestration, LangGraph routing, LangChain semantic retrieval, spaCy NER preprocessing, PII redaction, LoRA/DPO fine-tuning pipeline, and evidence-grounded recruiter Q&A. Built for hiring teams who deserve production-grade candidate intelligence without cloud API costs.*
>
> *Talentra leverages foundation model fine-tuning (LoRA/DPO on Mistral-7B/Phi-3-mini) for domain-adapted hiring intelligence — adapting large pre-trained language models to the recruitment domain via parameter-efficient fine-tuning.*

<br/>

**Built by [Akilan Manivannan](https://github.com/AkilanManivannanak)**

</div>

---

## 🎯 System Goal & SLOs

> **Goal:** Help recruiters screen candidates with grounded evidence while keeping the system cheap, stable, and locally runnable — zero external API spend by default.

All metrics benchmarked on MacBook M2 · local CPU · no external API calls · reproducible via `scripts/benchmark.py`

| SLO | Target | Achieved | Status |
|---|---|---|---|
| Evaluate p95 latency | < 1,500 ms | **4.81 ms** | ✅ 312× under target |
| Copilot Q&A p95 latency | < 1,500 ms | **4.41 ms** | ✅ 340× under target |
| Candidate upload p95 | < 500 ms | **18.76 ms** | ✅ |
| Role create p95 | < 100 ms | **6.46 ms** | ✅ |
| Top-1 evaluation accuracy | = 1.0 | **1.0** | ✅ |
| Copilot ranking consistency | = 1.0 | **1.0** | ✅ |
| External API cost / request | $0.00 | **$0.000** | ✅ |
| Error rate (benchmark suite) | 0% | **0%** | ✅ |

```
One-Liner for interviewers:
"Built multi-agent hiring intelligence with LangGraph + LangChain + spaCy
→ evaluate p95 4.81 ms · copilot p95 4.41 ms · cost $0.000/request · accuracy 1.0
→ FastAPI + Streamlit + Docker + Render · CI with 5 gates
→ 5 production bugs found, root-caused, fixed — all in the postmortem"
```

---

## 🎬 Demo

<div align="center">

| | |
|---|---|
| 🎥 **Video walkthrough** | [Full pipeline demo — role → upload → evaluate → copilot → ATS](https://drive.google.com/drive/folders/1Rx2Lzq3qjbHWWjdCU98pWKo9187kZzla?usp=sharing) |
| 📸 **Screenshots** | Role creation, resume upload, evaluation ranking, bias audit, Copilot Q&A, ATS workflow |
| 📊 **Benchmark JSON** | [`docs/benchmark_results.json`](docs/benchmark_results.json) — all SLO numbers, reproducible |

</div>

---

## 📐 Architecture & Data Flow

```
Resume Upload / Recruiter Query
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
│      role · upload · evaluate · copilot · ATS · ops        │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI serving layer                      │
│   X-Request-ID · /ops/metrics · /metrics · /health · CORS  │
└──────┬────────────────┬──────────────────┬──────────────────┘
       │                │                  │
       ▼                ▼                  ▼
┌──────────────┐ ┌─────────────┐  ┌──────────────────────┐
│ ① Preprocess │ │ ② LangChain │  │  ③ LangGraph         │
│   Pipeline   │ │    Layer    │  │     Orchestration    │
│              │ │             │  │                      │
│ clean text   │ │ PDF/DOCX    │  │  preprocess_node     │
│ detect sects │ │ loaders     │  │  embed_node          │
│ skill NER    │ │ recursive   │  │  evaluate_node       │
│ PII redact   │ │ splitter    │  │  route_question() ──▶│
│ tenure parse │ │ vectorstore │  │  copilot_qa_node     │
│              │ │ (Chroma /   │  │  evidence_node       │
│ spaCy NER    │ │ FAISS /     │  │  ats_update_node ◀── │
│ Presidio     │ │ lexical)    │  │  human-in-the-loop   │
└──────┬───────┘ └──────┬──────┘  └──────────┬───────────┘
       └────────────────┘                     │
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ④ Multi-Agent System                      │
│                                                             │
│  ScreenerAgent     → hard-filter on must-have requirements  │
│  RankerAgent       → evidence-weighted 0–3 score per req    │
│  InterviewerAgent  → tailored STAR + technical questions    │
│  BiasAuditorAgent  → responsible AI safety layer            │
│                      gender / prestige / recency audit      │
│  CopilotAgent      → evaluation-aware Q&A + evidence search │
│                                                             │
│  BaseAgent: _call_llm · _parse_json · _format_evidence      │
│  All agents degrade gracefully — rule-based fallback always  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               ⑤ LLM Fine-tuning Layer (offline)             │
│                                                             │
│  data_generator → SFT JSONL (fixtures + ATS feedback)       │
│  LoRATrainer    → LoRA/QLoRA · Mistral-7B / Phi-3-mini      │
│  DPOTrainer     → preference pairs from recruiter decisions  │
│  eval_gate      → benchmark.py --assert before promotion     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Storage                           │
│   metadata.json · data/vectorstore/ · data/finetune/        │
│   models/active_model.json · ATS stage + recruiter notes    │
└─────────────────────────────────────────────────────────────┘
```

### Ingest → Store → Retrieve → Infer → Feedback

| Stage | What happens | Technology |
|---|---|---|
| **Ingest** | Parse PDF/DOCX/TXT/MD, clean unicode, fix ligatures, strip PDF noise | LangChain loaders + PyPDF2 fallback |
| **Preprocess** | Detect sections, extract skills via NER, redact PII, parse tenure | spaCy + Presidio + regex |
| **Store** | Section-aware chunking, persist to vectorstore + metadata JSON | RecursiveCharacterTextSplitter + Chroma/FAISS/lexical |
| **Retrieve** | IDF-weighted lexical or semantic similarity search per requirement | TalentraVectorStore (auto-backend) |
| **Infer** | Screen → rank → audit → answer — all agents in LangGraph DAG | ScreenerAgent + RankerAgent + CopilotAgent |
| **Feedback** | Recruiter decisions → DPO preference pairs → fine-tune loop | ATS notes + data_generator + DPOTrainer |

---

## 🚀 Quick Start

### Option 1 — Local (Recommended)

```bash
# 1. Clone the repo
git clone git@github.com:AkilanManivannanak/talentra_copilot.git
cd talentra_copilot

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy NER model
python -m spacy download en_core_web_sm

# 5. Configure environment
cp .env.example .env

# 6. Start the API (Terminal 1)
uvicorn app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs      ← Swagger UI with all routes
# → http://localhost:8000/metrics   ← Prometheus scrape endpoint

# 7. Start the UI (Terminal 2)
streamlit run frontend/streamlit_app.py
# → http://localhost:8501
```

### Option 2 — Docker (One command)

```bash
cp .env.example .env
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required |
| spaCy model | `en_core_web_sm` | `python -m spacy download en_core_web_sm` |
| Docker | 24+ | Optional — containerised run |
| GPU / Apple Silicon | — | Optional — only needed for fine-tuning |

---

## 🔧 Tech Stack & Trade-offs

| Layer | Technology | Trade-off Rationale |
|---|---|---|
| **Orchestration** | LangGraph 0.1+ | Stateful DAG + conditional routing; `interrupt_before` for human-in-the-loop ATS writes |
| **API** | FastAPI 0.115+ | Async, typed, auto-docs; `X-Request-ID` + per-route latency middleware |
| **UI** | Streamlit 1.45+ | Rapid recruiter-facing iteration; full 6-page agent loop in one file |
| **Document loading** | LangChain loaders | `PyPDFLoader` + `Docx2txtLoader`; PyPDF2/python-docx fallback |
| **Text splitting** | `RecursiveCharacterTextSplitter` | Section-aware metadata tagging — EXPERIENCE/SKILLS chunks retrieved separately |
| **Vector store** | Chroma → FAISS → lexical | Auto-selected at startup; zero-infra lexical fallback for Mac/CI |
| **NER** | spaCy `en_core_web_sm` | Local, fast, no API cost; 50-skill curated taxonomy; regex fallback |
| **PII redaction** | Microsoft Presidio | Email, phone, SSN, URLs; regex fallback when Presidio not installed |
| **Fine-tuning** | `peft` + `trl` (LoRA/DPO) | Runs on consumer GPU; 4-bit QLoRA via bitsandbytes on CUDA |
| **Observability** | Prometheus + custom `/ops/metrics` | `/metrics` for infra scraping; `/ops/metrics` for human-readable latency dashboard |
| **Storage** | JSON flat files | Zero infra; suitable for < 10k candidates; swap to Postgres in one line |
| **Container** | Docker + Compose | One-command reproducible deploy; spaCy model baked into image |
| **Cloud deploy** | Render.com (`render.yaml`) | Free tier; separate services for API + UI |

**Key architectural decisions:**

* **Local-first, $0 default** — zero OpenAI API cost; LLM is an optional plug-in. Every agent has a deterministic fallback.
* **Lexical before dense** — IDF scoring is deterministic, fast, and debuggable. Upgrade path to Chroma semantic search is one config flag.
* **Flat JSON over Postgres** — sufficient for portfolio/demo scale; avoids infra dependency. The `MetadataStore` interface swaps trivially.
* **Graceful degradation at every layer** — spaCy absent → regex NER; Presidio absent → regex PII; Chroma absent → FAISS → lexical; LLM absent → rule-based agents.

---

## 📏 Latency Budget

> Where does latency hide? Tokenization, retrieval, rerank, LLM generation. Here's every stage measured.

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| Text cleaning + normalization | ~2 ms | ~6 ms | Pure Python, no I/O |
| spaCy NER + skill extraction | ~4 ms | ~12 ms | Local model, no API |
| PII redaction (Presidio) | ~3 ms | ~8 ms | In-process |
| Section-aware chunking | ~2 ms | ~5 ms | RecursiveCharacterTextSplitter |
| Vectorstore indexing | ~3 ms | ~8 ms | Lexical append to JSON |
| **Total upload (benchmark)** | **~16 ms** | **~19 ms** | All preprocessing + index |
| ScreenerAgent (must-have filter) | ~0.5 ms | ~1.5 ms | Regex term match |
| RankerAgent (IDF scoring, 3 candidates × 8 reqs) | ~1.5 ms | ~3 ms | No LLM call |
| BiasAuditorAgent | ~0.3 ms | ~1 ms | Heuristic checks |
| **Total evaluate (benchmark)** | **~1.8 ms** | **~4.8 ms** | All agents, 3 candidates |
| CopilotAgent routing + retrieval | ~1.2 ms | ~2 ms | Rule-based |
| **Total copilot (benchmark)** | **~1.6 ms** | **~4.4 ms** | Including evidence retrieval |
| **LLM path (if configured)** | ~300 ms | ~1,200 ms | Ollama llama3.1:8b local |

> **Where latency hides in production:** For the LLM-enabled path, 90% of latency is generation. Mitigation: semantic caching (cosine sim ≥ 0.92 → skip LLM), small→big model routing (rule-based screener → LLM ranker only for shortlist), and `top_k` reduction via ScreenerAgent.

---

## 🤖 Multi-Agent System

### Agent responsibilities

| Agent | Input | Output | LLM path | Rule-based fallback |
|---|---|---|---|---|
| `ScreenerAgent` | Candidates + must-have requirements | `{pass, fail_reasons}` per candidate | Yes/no single-turn judgement | Regex key-term matching |
| `RankerAgent` | Shortlist + requirements + vectorstore | Ranked list with 0–3 score per requirement | `EVALUATION_PROMPT` → JSON | IDF lexical scoring |
| `InterviewerAgent` | Role + candidate skills + top evidence | `[{type, skill_focus, question}]` | `INTERVIEW_QUESTION_PROMPT` | Curated STAR templates |
| `BiasAuditorAgent` | Ranked candidates + raw texts | `{bias_flags, severity, recommendation}` | `BIAS_AUDIT_PROMPT` | Name/school/gap heuristics |
| `CopilotAgent` | Question + eval results + vectorstore | Grounded answer string | `COPILOT_PROMPT` | Rule-based routing (comparison / evidence / generic) |

### 🛡️ Responsible AI Safety Layer

`BiasAuditorAgent` is Talentra's built-in responsible AI safety layer — automated bias detection for gender, prestige, and recency discrimination in hiring decisions. Every evaluation run triggers a bias audit across three dimensions:

| Dimension | What it checks | Signal |
|---|---|---|
| **Gender bias** | Name-based gender inference correlated with score deltas | Candidate score penalised by inferred gender |
| **Prestige bias** | FAANG/Ivy school names correlated with ranking uplift | Overweighting brand names over demonstrated skills |
| **Recency bias** | Short tenure or employment gaps correlated with rejection | Penalising career breaks, not skill gaps |

Output: `{bias_flags: [...], severity: low/medium/high, recommendation: str}` — surfaced to the recruiter before ATS stage promotion. Audits run deterministically with a rule-based fallback — no LLM required in the critical safety path.

> *Safety in hiring AI is not a feature — it's a constraint. BiasAuditorAgent runs on every evaluation, not on request.*

### LangGraph workflow

```
preprocess_node  →  embed_node  →  evaluate_node
                                        │
                              [route_question()]
                               ↙              ↘
                         copilot_qa      evidence_search
                               ↘              ↙
                            ats_update_node
                         ⚠️  interrupt fires here
                         recruiter approves/cancels
                               ↓
                              END
```

**Cost + quality routing:** `route_question()` detects comparison keywords ("compare", "ranked above", "strongest") and routes to `copilot_qa` (uses evaluation summary). Skill-specific questions route to `evidence_search` (targeted retrieval, no eval context). This prevents the Copilot from contradicting the ranker — a real bug we fixed (see postmortem v4→v5).

---

## 🔁 Reliability — Fallbacks & Observability

```
Upload Request
      │
      ▼
┌──────────────────────┐  error  ┌──────────────────────┐
│  LangChain loaders   │ ──────▶ │  PyPDF2 / python-    │
│  (primary)           │         │  docx (fallback)     │
└──────────┬───────────┘  ok    └──────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│                 Preprocessing pipeline               │
│  spaCy NER  ──fail──▶  regex taxonomy fallback       │
│  Presidio   ──fail──▶  regex PII patterns            │
│  Chroma     ──fail──▶  FAISS  ──fail──▶  lexical     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│         Partial-success batch upload                 │
│  1 bad PDF in a 4-file batch → structured error      │
│  other 3 files process normally                      │
│  failed ingest → candidate record cleaned up         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│            Observability layer                       │
│  X-Request-ID on every response                      │
│  GET /ops/metrics  → p50/p95/p99 per route (custom)  │
│  GET /metrics      → Prometheus scrape endpoint      │
│  GET /ops/build    → env, model, started_at          │
│  GET /health       → status check for load balancers │
└──────────────────────────────────────────────────────┘
```

**Failure modes handled:**

| Failure | Response |
|---|---|
| 1 bad PDF in batch | Structured error per file; rest process normally |
| spaCy not installed | Skill extraction falls back to regex n-gram scan |
| Presidio not installed | PII redaction falls back to 7 regex patterns |
| Chroma/FAISS not installed | Vectorstore falls back to `data/vectorstore/index.json` |
| LLM not configured | All 5 agents use deterministic rule-based fallbacks |
| Vectorstore empty | CopilotAgent answers from evaluation summary only |
| ATS update pending | LangGraph `interrupt_before` pauses for recruiter approval |

---

## 🛠 Preprocessing Pipeline

Every uploaded resume goes through 5 stages before a single chunk hits the vector store:

```
raw text
   │
   ▼  clean_text()
   │  unicode normalize · ligature fix (ﬁ→fi) · whitespace collapse · PDF noise strip
   │
   ▼  detect_sections()
   │  regex → EXPERIENCE / SKILLS / EDUCATION / REQUIREMENTS / CONTACT / PROJECTS
   │  section label attached to every downstream chunk as metadata
   │
   ▼  extract_skills()
   │  spaCy PRODUCT/ORG/LANGUAGE entities → 50-skill canonical taxonomy
   │  regex n-gram scan (1–4 tokens) → alias map → canonical names
   │  fallback: pure regex when spaCy not installed
   │
   ▼  redact_pii()
   │  Presidio → EMAIL_ADDRESS / PHONE_NUMBER / US_SSN / URL / PERSON → <PLACEHOLDER>
   │  fallback: 7 compiled regex patterns (email, phone, SSN, ZIP, LinkedIn, GitHub, URL)
   │
   ▼  parse_tenure()
      "Jan 2020 – Present" → {start, end, duration_years}
      total_years_experience() = Σ all non-overlapping ranges
```

**Real output on an actual resume:**

```
Skills:   ['docker', 'fastapi', 'langchain', 'machine learning', 'python', 'rag']
PII:      ['PERSON', 'EMAIL_ADDRESS', 'DATE_TIME', 'URL']
Sections: ['CONTACT', 'EXPERIENCE', 'SKILLS', 'EDUCATION']
YOE:      6.3 years
```

---

## 🧪 Evaluation Gates

| Metric | Tool | Gate | Action if Fail |
|---|---|---|---|
| Top-1 evaluation accuracy | `scripts/benchmark.py` fixture | `= 1.0` | Block PR merge via CI |
| Copilot ranking consistency | `scripts/benchmark.py` fixture | `= 1.0` | Block PR merge via CI |
| Evaluate p95 | `benchmark.py --assert` | `≤ 1,500 ms` | Block PR merge via CI |
| Copilot p95 | `benchmark.py --assert` | `≤ 1,500 ms` | Block PR merge via CI |
| Fine-tuned model quality | `app/finetuning/eval_gate.py` | all above pass | Block model promotion to `models/active_model.json` |
| Module syntax | `python -m py_compile` (CI job 1) | no errors | Block PR merge |
| Preprocessing smoke | custom assertions (CI job 3) | skills + PII detected | Block PR merge |
| Training data schema | JSONL assertion (CI job 4) | 3-message format valid | Block PR merge |

---

## 🔮 Fine-tuning Pipeline (LLM upgrade path)

```
eval/demo_seed.json ──┐
ATS recruiter feedback  ──▶  data_generator.py ──▶  train.jsonl   (SFT)
Copilot interaction logs ─┘                    └──▶  dpo_pairs.jsonl (DPO)
                                                           │
                                          ┌────────────────┴─────────────────┐
                                     LoRATrainer                       DPOTrainer
                                   (peft + trl)                      (trl DPO)
                                 Phi-3-mini / Mistral-7B         beta=0.1
                                 r=16, alpha=32, dropout=0.05    recruiter prefs
                                 4-bit QLoRA (CUDA)               ↕
                                          └────────────────┬─────────────────┘
                                                     eval_gate.py
                                                  benchmark.py --assert
                                                           │
                                              pass ────────┴──────── fail
                                               │                        │
                                  models/active_model.json           blocked
                               (serving layer picks this up)    (stays on rule-based)
```

```bash
# 1. Install fine-tuning deps
pip install transformers>=4.40 peft>=0.10 trl>=0.8 datasets>=2.18 accelerate>=0.29
# GPU only: pip install bitsandbytes>=0.43

# 2. Generate training data (works immediately — no GPU needed)
python scripts/generate_training_data.py \
  --seed eval/demo_seed.json \
  --out data/finetune/train.jsonl
# ✓ Wrote 12 training examples to data/finetune/train.jsonl

# 3. LoRA fine-tuning (--no-4bit for Mac; omit on CUDA)
python scripts/finetune_lora.py \
  --data data/finetune/train.jsonl \
  --base-model microsoft/phi-3-mini-4k-instruct \
  --no-4bit --epochs 3
# → models/talentra-lora/final_adapter/

# 4. DPO from recruiter feedback
python scripts/finetune_dpo.py \
  --model models/talentra-lora/final_adapter
# → models/talentra-dpo/final/

# 5. Eval gate (runs automatically; also callable standalone)
make eval-gate
```

**MLOps awareness:**
* Eval-gate-before-promote — fine-tuned model can only become active if it passes all SLO assertions
* Rollback path — delete `models/active_model.json` to revert to rule-based instantly
* Shadow testing — run fine-tuned and rule-based in parallel; compare outputs before cutover
* Drift detection — RAGAS scores logged per interaction for offline quality drift monitoring

---

## 📉 Postmortem Log

> *5 production bugs found, root-caused, and fixed. All documented.*

| Version | Issue | Root Cause | Fix Applied |
|---|---|---|---|
| v1→v2 | OpenAI quota failures blocked all ingestion | Embeddings hard-coupled to OpenAI API; one quota failure = total outage | Replaced default path with local lexical retrieval; external API is opt-in only |
| v2→v3 | One bad PDF crashed entire upload batch | Upload processing was all-or-nothing; no per-file error isolation | Per-file try/except; partial-success response with structured `{errors: []}` array |
| v3→v4 | Ghost candidates after failed ingest | `MetadataStore.create_candidate()` ran before vectorstore indexing completed | Candidate record cleanup runs on indexing failure; orphans removed automatically |
| v4→v5 | Copilot contradicted evaluation ranking | Comparison questions hit raw lexical search instead of evaluation results | `route_question()` in LangGraph detects comparison keywords; routes to `copilot_qa_node` which reads eval summary first |
| v5→v6 | Resume header/contact junk polluted evidence | Email, phone, LinkedIn chunks indexed at same weight as experience text | Contact-noise IDF penalty (×0.3) + `redact_pii()` before indexing + evidence deduplication |

---

## 🗂 Project Structure

```
talentra_copilot/
├── app/
│   ├── main.py                     # FastAPI entry point — all routes, CORS, Prometheus, middleware
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py               # Pydantic Settings — all config from .env, @lru_cache
│   │   ├── store.py                # MetadataStore — thread-safe JSON persistence
│   │   ├── middleware.py           # X-Request-ID + RequestMetrics (p50/p95/p99 per route)
│   │   └── requirements.py         # JD requirement extractor (regex + fallback)
│   ├── routers/
│   │   ├── roles.py                # POST /roles/text · GET /roles/ · GET /roles/{id}
│   │   ├── candidates.py           # POST /candidates/upload · GET · DELETE
│   │   ├── evaluation.py           # POST /roles/{id}/evaluate (all 3 agents)
│   │   ├── copilot.py              # POST /copilot/query · /copilot/interview-questions
│   │   ├── ats.py                  # GET /ats/candidates · PATCH stage · POST notes
│   │   └── ops.py                  # GET /health · /ops/metrics · /ops/build
│   ├── preprocessing/
│   │   ├── cleaner.py              # Unicode normalize, ligature fix, PDF noise strip
│   │   ├── sections.py             # Regex section detector (10 section types)
│   │   ├── skills.py               # spaCy NER + 50-skill taxonomy alias map
│   │   ├── pii.py                  # Presidio redaction + 7 regex fallback patterns
│   │   ├── dates.py                # "Jan 2020–Present" parser, YOE calculator
│   │   └── pipeline.py             # run_preprocessing_pipeline() → PreprocessedDocument
│   ├── langchain_layer/
│   │   ├── loaders.py              # LangChain loaders with PyPDF2/docx fallback
│   │   ├── splitter.py             # RecursiveCharacterTextSplitter + section metadata
│   │   ├── vectorstore.py          # TalentraVectorStore: Chroma→FAISS→lexical auto-select
│   │   └── prompts.py              # EVALUATION / COPILOT / INTERVIEW / BIAS prompt templates
│   ├── graph/
│   │   └── hiring_graph.py         # LangGraph StateGraph — 6 nodes + conditional router
│   ├── agents/
│   │   ├── base.py                 # BaseAgent: _call_llm · _parse_json · _format_evidence
│   │   ├── screener.py             # ScreenerAgent — must-have hard filter
│   │   ├── ranker.py               # RankerAgent — 0–3 evidence score per requirement
│   │   ├── interviewer.py          # InterviewerAgent — STAR + technical questions
│   │   ├── bias_auditor.py         # BiasAuditorAgent — responsible AI safety layer
│   │   │                           #   gender / prestige / recency discrimination audit
│   │   └── copilot.py              # CopilotAgent — eval-aware Q&A + evidence search
│   └── finetuning/
│       ├── data_generator.py       # SFT JSONL from fixtures + ATS feedback
│       ├── trainer.py              # LoRATrainer — peft + trl, 4-bit QLoRA support
│       ├── dpo.py                  # DPOTrainer — preference training from recruiter votes
│       └── eval_gate.py            # Benchmark gate; writes models/active_model.json on pass
├── frontend/
│   └── streamlit_app.py            # Full recruiter UI — 6 pages
├── scripts/
│   ├── benchmark.py                # Reproducible benchmark + SLO assertions (--assert flag)
│   ├── seed_demo.py                # Seed Akila/Esha/Jaxon demo candidates
│   ├── generate_training_data.py   # CLI: generate SFT JSONL from seed + feedback
│   ├── finetune_lora.py            # CLI: LoRA fine-tuning with eval gate
│   └── finetune_dpo.py             # CLI: DPO fine-tuning from preference pairs
├── eval/
│   └── demo_seed.json              # Ground-truth fixture: Akila > Esha > Jaxon
├── docs/
│   ├── architecture.md             # Full architecture documentation
│   ├── benchmark_results.json      # Latest benchmark run output
│   ├── benchmark.md                # Benchmark summary (auto-generated)
│   └── postmortem.md               # 5-incident production postmortem log
├── .github/
│   └── workflows/ci.yml            # 5 CI jobs: lint · test · preproc smoke · data smoke · benchmark
├── Dockerfile                      # python:3.12-slim + spaCy model at build time
├── docker-compose.yml              # backend + frontend services
├── Makefile                        # 15 targets: setup · test · benchmark · finetune · docker
├── render.yaml                     # Render.com deploy: separate API + UI web services
├── requirements.txt                # Pinned deps; fine-tuning section commented out
├── .env.example                    # All config keys documented
└── README.md
```

---

## ⚙️ Makefile — All Targets

```bash
make setup-all        # venv + pip install -r requirements.txt + spaCy download
make setup-finetune   # pip install transformers peft trl datasets bitsandbytes accelerate
make run-api          # uvicorn app.main:app --reload  → :8000
make run-ui           # streamlit run frontend/streamlit_app.py  → :8501
make test             # pytest -q
make test-cov         # pytest --cov=app --cov-report=term-missing
make lint             # python -m py_compile all 40 modules
make benchmark        # scripts/benchmark.py → docs/benchmark_results.json
make smoke            # test + benchmark + --assert (full CI gate locally)
make preprocess-demo  # quick preprocessing smoke (skills/PII/YOE printed)
make generate-data    # → data/finetune/train.jsonl
make finetune-lora    # LoRA training (needs GPU deps)
make finetune-dpo     # DPO training (needs GPU deps)
make eval-gate        # standalone eval gate check
make docker-up        # docker compose up --build
make docker-down      # docker compose down
```

---

## 🔬 CI / CD & MLOps

### CI pipeline (`.github/workflows/ci.yml`) — 5 parallel jobs

| Job | What it checks | Gate |
|---|---|---|
| **lint** | `py_compile` all 40 Python modules | Syntax error → fail |
| **test** | `pytest --cov=app` | Test failure → fail |
| **preprocessing-smoke** | clean/NER/PII/dates without spaCy | Skills + PII not detected → fail |
| **training-data-smoke** | JSONL generation from fixture | Schema invalid → fail |
| **benchmark** | `benchmark.py --assert` | p95 or accuracy below threshold → fail |

### Observability

```
Every request → X-Request-ID header
             → X-Response-Time-Ms header
             → recorded in RequestMetrics (in-memory p50/p95/p99 per route)

GET /ops/metrics  → custom in-memory latency summary per route (human-readable JSON)
GET /ops/build    → app_env, python version, model backend, started_at
GET /health       → {"status": "ok"}  ← used by Render load balancer
GET /metrics      → Prometheus scrape endpoint (machine-readable text format)
                    exposes: http_requests_total · http_request_duration_seconds
                             http_requests_in_progress · labelled by route/method/status
```

**Dual observability design — intentional:**

| Endpoint | Format | Purpose | Consumer |
|---|---|---|---|
| `GET /ops/metrics` | JSON | Human-readable latency dashboard | Recruiter ops page, Streamlit UI |
| `GET /metrics` | Prometheus text | Machine-readable scrape target | Prometheus server, Grafana Cloud |

The two endpoints serve different consumers and are not redundant. `/ops/metrics` is the application-level dashboard; `/metrics` is the infrastructure monitoring scrape endpoint.

### Cloud deploy (Render.com)

```yaml
# render.yaml — two separate web services
talentra-api:  uvicorn app.main:app  (healthCheckPath: /health)
talentra-ui:   streamlit run frontend/streamlit_app.py
```

### Rollback path

```bash
# Revert fine-tuned model → rule-based instantly
rm models/active_model.json

# Wipe vectorstore and re-ingest
rm -rf data/vectorstore/
uvicorn app.main:app --reload
# → upload resumes again; fresh index built
```

---

## 💬 Demo Flow

1. **Create Role** → paste any job description → requirements auto-extracted from JD text
2. **Upload Resumes** → PDF / DOCX / TXT → preprocessing pipeline fires automatically:
   - skills detected, PII redacted, sections labelled, tenure parsed
3. **Evaluate** → ScreenerAgent drops must-have failures → RankerAgent scores 0–3 per requirement → BiasAuditorAgent runs responsible AI safety audit (gender/prestige/recency)
4. **Copilot Q&A** → ask in plain English:
   - *"Who is the strongest candidate for this role?"*
   - *"Why is the top candidate ranked first?"*
   - *"Compare all candidates against the role requirements."*
   - *"Who shows the strongest evidence for machine learning?"*
5. **Interview Questions** → generate tailored STAR + technical questions for any shortlisted candidate
6. **ATS Workflow** → move candidates: `new → screening → shortlisted → interview → offer → hired/rejected` · add recruiter notes
7. **Ops Dashboard** → real-time p50/p95/p99 latency per route + Prometheus metrics at `/metrics`

---

## 🗣 Interview-Ready Answers

> *Practised for system design interviews — reproduced here for transparency.*

**"Design a RAG for 1M PDFs; latency < 1.5s — where do caching and rerankers live?"**
> Semantic cache (cosine ≥ 0.92) sits in front of the LangGraph pipeline — cache HIT bypasses LLM entirely (~120ms). Reranker (BM25 term boost) sits between ChromaDB MMR retrieval and generation — precision↑ at low K, adds ~15ms p95. For 1M PDFs: distributed vectorstore (Weaviate/OpenSearch), async batch ingest, HNSW index. Talentra uses the same architecture pattern at portfolio scale.

**"Deploy an LLM assistant with small→big model routing, cost guardrails, fail-open"**
> ScreenerAgent = cheap rule-based model (0 cost). RankerAgent = local Phi-3-mini via LoRA adapter (0 API cost). CopilotAgent = Ollama llama3.1:8b for complex Q&A. Cost guardrail: token counter per request; switch to smaller model tier if daily budget exceeded. Fail-open: every agent has a rule-based fallback — LLM unavailable → rule-based answer, never a 500 error.

**"Make it resilient to data drift — eval gates, rollbacks, shadow tests"**
> Eval gate: `benchmark.py --assert` blocks fine-tuned model promotion if accuracy drops. Rollback: delete `models/active_model.json` → rule-based backend instantly. Shadow test: run fine-tuned and rule-based in parallel; compare `pct_score` distributions before cutover. Drift signal: RAGAS faithfulness logged per interaction — dashboard alert if rolling average drops below 0.70.

---

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn)

**© 2026 Akilan Manivannan — All Rights Reserved**

*Talentra Copilot · LangGraph · LangChain · spaCy · Presidio · FastAPI · Streamlit · Docker · Render · Prometheus*

[![GitHub](https://img.shields.io/badge/GitHub-AkilanManivannanak-181717?style=flat-square&logo=github)](https://github.com/AkilanManivannanak)

</div>
