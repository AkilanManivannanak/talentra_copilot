# Talentra Copilot

Evidence-grounded candidate screening. Paste a job description, upload resumes, get a
ranking where every score traces back to a quoted line in a resume — plus a measurement of
whether the assistant's own answers are actually supported by the evidence it cites.

Runs entirely on local CPU. No API keys, no external calls, **$0.00 per request** at every
retrieval mode.

```bash
make setup          # installs deps + the spaCy model
make run-api        # http://localhost:8000/docs
make eval           # nDCG@5 / P@3 / MRR against graded relevance judgements
make benchmark      # cold and warm latency, reported separately
```

---

## What it does

| | |
|---|---|
| **Extract** | Segment a JD into clause-level requirements, flag the must-haves, weight the rest |
| **Ingest** | Parse PDF/DOCX/TXT → clean → detect sections → extract skills (spaCy NER + 30-skill taxonomy) → **redact PII before indexing** (Presidio) → parse tenure |
| **Retrieve** | Lexical IDF + phrase scoring, fused with dense retrieval via reciprocal rank fusion; optional cross-encoder rerank |
| **Rank** | Per-requirement scores with cited evidence, weighted by must-have status |
| **Screen** | Pool-relative hard filter on must-haves, before ranking |
| **Answer** | Recruiter Q&A routed through the evaluation for comparison questions, through retrieval for factual ones |
| **Verify** | Every answer scored sentence-by-sentence for groundedness against its own citations |
| **Audit** | Counterfactual bias audit: re-score each resume with school / employer / name redacted and report the delta |
| **Act** | ATS stage transitions and notes, behind a human-in-the-loop approval gate that genuinely blocks the write |

---

## Retrieval quality

Measured against `eval/labels.json`: **30 candidates, 3 roles, 15 requirements, 180 graded
relevance judgements** on the standard 0–3 scale. Each requirement is a query; candidates
are the documents. Reproduce with `make eval-ablation`.

| Configuration | nDCG@5 | P@3 | Recall@5 | MRR | Kendall τ | p95 (ms) |
|---|---|---|---|---|---|---|
| lexical (IDF + phrase) | 0.761 | 0.911 | 0.454 | 0.933 | 0.249 | 3.4 |
| **dense** (LSA, 192-d) | **0.799** | 0.911 | 0.456 | **1.000** | **0.288** | 3.0 |
| hybrid (RRF, k=60) | 0.780 | **0.933** | 0.430 | **1.000** | 0.240 | 13.4 |
| hybrid + rerank | 0.780 | **0.933** | 0.430 | **1.000** | 0.240 | 7.3 |

**Hybrid is not uniformly best, and that is reported rather than tuned away.** Dense wins
nDCG@5 by 1.9 points; hybrid wins precision@3 and ties on MRR. The default is `hybrid`
because surfacing the right people in the top three is the job a recruiter actually does
with this tool, and P@3 is the metric for that. Dense would be the right default if the
goal were whole-list ordering.

Reranking shows no quality change here because the cross-encoder requires the optional ML
extra and a model download; the row is the same configuration with the stage inactive, and
`/ops/build` reports `rerank_enabled: false` accordingly. It is listed so the ablation
table matches the code paths that exist, not to imply a result that was not measured.

**Where it fails.** The harness prints the worst queries every run, because an eval you
only read the average of is a scoreboard rather than a diagnostic. Current worst:

```
0.416  Nice to have: experience with retrieval systems or vector search
0.609  Strong Python skills for research and production code
0.663  Must have strong Python engineering skills in a production codebase
```

The pattern is consistent: requirements whose discriminating vocabulary is common across
the corpus ("Python", "production") rank poorly, because neither retriever can tell
*having* a capability from *mentioning* it. That is the next thing worth fixing, and it is
the same root cause as the negation problem documented in `ScreenerAgent`.

### Honest limitations of this eval set

- The resumes are **synthetic**, authored from structured evidence profiles
  (`eval/build_labels.py`). Using real resumes would mean processing other people's
  personal data to publish a benchmark.
- Judgements are derived from those profiles, never from the prose. The prose uses
  paraphrases, so the surface form does not hand the answer to a lexical matcher — a test
  asserts phrasings stay varied.
- Synthetic prose is cleaner than real resumes, so **absolute scores here are optimistic**.
  Use the set for relative comparison between configurations, which is what the ablation
  does.

---

## Answer groundedness

A citation list next to an answer proves nothing about whether the citations support what
was said. Every Copilot answer is scored:

```
faithfulness = supported_claim_sentences / claim_sentences
```

A sentence is supported when at least one cited chunk covers it — cosine similarity when a
dense backend is loaded, IDF-weighted content-term coverage otherwise. The backend used is
reported with the score.

**Current measured value on the benchmark question: 0.33** (1 of 3 sentences supported).
This is a real finding, not a rounding error. Ranking answers restate figures that come
from the evaluation — *"The next closest candidate is X at 0.20"* — and those figures are
not in the resume text the citations point at. The metric is correctly calling that
ungrounded. The fix is to cite the evaluation as a source alongside the resume chunks,
which is open work; publishing 0.33 and naming the cause beats not measuring it.

---

## Latency

In-process FastAPI `TestClient`, single machine, no concurrency, 12 candidates, 6
requirements, n=200 per measurement. These are **benchmark numbers, not SLOs** — an SLO is
a commitment about production traffic.

| Operation | p50 | p95 | p99 |
|---|---|---|---|
| evaluate (cold) | 9.9 ms | 12.5 ms | 29.2 ms |
| evaluate (warm — cache hit) | 4.5 ms | 4.8 ms | 4.9 ms |
| copilot (cold) | 11.2 ms | 17.1 ms | 37.7 ms |
| copilot (warm — cache hit) | 6.0 ms | 6.7 ms | 7.2 ms |
| candidate upload (batch of 12) | — | 2.23 s | — |
| role create | — | 6.5 ms | — |

Cold and warm are separated because `RankingService` memoises on
`(role, candidates, k, store versions)`. **A previously published figure of "evaluate p95
4.81 ms" was a warm number**: the old benchmark fired the identical request 20 times with
no writes in between, so 19 of 20 samples were cache hits and the p95 index landed on one
of them. Cache speed-up on evaluate p95 is 2.6×.

Preprocessing model construction (spaCy + Presidio) is a **boot cost of roughly 8–40 s
depending on the machine**, paid once on a background thread at startup, reported
separately in `docs/benchmark_results.json`. Charging it to whichever request arrived first
is what made batch upload look like a 10-second operation.

---

## Architecture

```
                      POST /roles/text · /candidates/upload
                                     │
            ┌────────────────────────▼────────────────────────┐
            │  Ingest: parse → clean → sections → skills      │
            │  → PII redaction → tenure                        │
            │  (the REDACTED text is what gets indexed)        │
            └────────────────────────┬────────────────────────┘
                                     ▼
            ┌─────────────────────────────────────────────────┐
            │  HybridRetriever                                 │
            │    lexical: IDF + phrase + contact-noise penalty │
            │    dense:   LSA (default) or bge-small (opt-in)  │
            │    fusion:  reciprocal rank fusion, k=60         │
            │    rerank:  cross-encoder (opt-in)               │
            │  search_grouped() — one corpus-wide retrieval,   │
            │  bucketed by candidate                           │
            └────────────────────────┬────────────────────────┘
                                     ▼
   ┌─────────────────────┬───────────┴──────────┬──────────────────────┐
   │  RankingService     │  CopilotService      │  CounterfactualBias  │
   │  per-requirement    │  eval-aware routing  │  Auditor             │
   │  evidence + weights │  + FaithfulnessScorer│  redact → re-score   │
   └─────────────────────┴──────────┬───────────┴──────────────────────┘
                                     │
   POST /roles/{id}/evaluate/agentic │  LangGraph, bound to the SAME services
                                     ▼
        screen ─► evaluate ─► audit_bias ─► route_question ─┬─► copilot_qa
                                                            └─► evidence_search
                                     │
                            ┌────────▼────────┐
                            │  ⛔ INTERRUPT   │  checkpointed; returns with the
                            │   ats_update    │  write NOT applied
                            └────────┬────────┘
                                     │ POST /agentic/runs/{id}/resume {approve:true}
                                     ▼
                              ATS write committed
```

The graph nodes call the same services the REST endpoints call. They are built by
`build_hiring_graph(services)` and close over the live container, so the agentic path and
the plain path cannot produce different rankings for the same input.

The interrupt is real: `graph.compile(checkpointer=..., interrupt_before=["ats_update"])`.
`tests/test_agentic.py::test_ats_write_is_blocked_until_a_human_approves` asserts the
candidate's stage is unchanged after the run returns and only changes after approval.

---

## Responsible AI: what the bias audit does and does not do

**It performs no demographic inference.** An earlier version classified candidates by
matching first names against a hardcoded 29-name gender list and flagged group-mean gaps
computed over as few as one candidate per group. That is the mechanism responsible-AI
tooling exists to prevent — it fabricates a protected attribute the system was never given,
and it fails hardest on names outside the list it was written from. It has been removed,
and `tests/test_bias_audit.py` fails the build if it returns.

What runs instead measures the system's own behaviour:

```
delta_school   = score(resume) − score(resume with school names redacted)
delta_employer = score(resume) − score(resume with employer brands redacted)
delta_name     = score(resume) − score(resume with the person's name redacted)
```

A large **positive** delta means the ranking is rewarding a brand rather than the work. A
large **negative** delta means redaction destroyed real evidence, which is a retrieval bug
worth knowing about. Each variant is scored in a throwaway index so IDF is not contaminated
by the original document. The audit also reports requirements that matched *no* candidate —
usually an extraction failure, and the most common source of silent unfairness.

---

## Configuration

Everything is environment-driven; see `.env.example`. The settings that change behaviour:

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | `lexical` \| `dense` \| `hybrid` |
| `EMBEDDING_MODEL` | `lsa` | LSA needs only scikit-learn and downloads nothing. Set to `BAAI/bge-small-en-v1.5` after `pip install -r requirements-ml.txt` |
| `RERANK_ENABLED` | `false` | Cross-encoder stage; requires the ML extra |
| `REDACT_PII_ON_INGEST` | `true` | Redacted text is what reaches the index |
| `AGENTIC_INTERRUPT_BEFORE_ATS` | `true` | Set false only if you want unattended ATS writes |

`GET /ops/build` reports what the running process can actually do — dense backend, spaCy,
Presidio and LangGraph availability are **probed at runtime**, so the deployment cannot
drift away from this README.

---

## Graceful degradation

Every optional component has a fallback, and the active configuration is always
observable at `/ops/build` rather than assumed:

| Missing | Behaviour |
|---|---|
| scikit-learn | dense unavailable → `hybrid` reports itself as `lexical` |
| sentence-transformers | `EMBEDDING_MODEL=lsa` still gives dense retrieval |
| spaCy model | skill extraction falls back to the regex taxonomy |
| Presidio | PII redaction falls back to 7 regex patterns |
| langgraph | sequential executor honouring the same interrupt contract, reported as `langgraph_available: false` |

---

## Testing and CI

```
make lint      # ruff
make test      # 58 tests, 68% coverage
make eval      # retrieval quality gate
make benchmark # latency regression gate
```

Five CI jobs, gated in order: **lint → tests → (retrieval eval ‖ latency benchmark ‖ docker
build + healthcheck)**.

`pytest` runs first and blocks everything downstream. It was previously absent from CI
entirely — removed in commit `9a5d2b4` ("robust CI — remove pytest dependency") — and the
suite had been red on `main` ever since, which is how a `/ops/metrics` endpoint returning
`{"metrics": {}}` shipped alongside a passing build badge. The test that caught it existed
the whole time.

The retrieval job rebuilds `eval/labels.json` and fails if the committed file drifts from
its generator, then asserts nDCG@5 has not regressed below a floor.

---

## Known limitations

Listed because they are the questions worth asking, not because they are resolved.

1. **Groundedness is 0.33 on ranking answers.** Cause identified above: score figures come
   from the evaluation, not the cited resume text. Fix is to cite the evaluation as a source.
2. **Neither retriever handles negation.** "No production ML systems" scores similarly to
   "shipped production ML systems". The screener works around it by comparing candidates
   against each other instead of against a threshold; the ranker does not.
3. **The eval set is synthetic.** See the limitations section above.
4. **Storage is a JSON file.** Writes are atomic (`os.replace`) and guarded by a lock, and
   lookups are indexed, but it is single-process: the caches are per-worker, so two uvicorn
   workers can serve different cached answers. SQLite is the honest next step.
5. **No concurrency measurement.** Every latency number here is single-threaded.
6. **The LoRA pipeline is offline and not wired into serving.** `SummaryService` is
   rule-based and loads no causal LM. The eval gate compares an adapter against its base
   model on groundedness and refuses to promote without that comparison, but nothing in the
   request path consumes the result. `docs/training_results.md` states this.

---

## Repository

```
app/
  core/          config, logging, observability middleware
  models/        pydantic schemas (the API contract)
  preprocessing/ clean · sections · skills · PII · tenure  (runs on every upload)
  services/      the live stack — retrieval, ranking, copilot, faithfulness, bias audit
  agents/        screener · bias auditor · interviewer   (ranking and Q&A are services,
                 so the agentic and REST paths share one implementation)
  graph/         LangGraph hiring workflow
  routers/       roles · candidates · copilot · documents · ats · agentic · ops
  finetuning/    offline LoRA/DPO pipeline + model-in-the-loop eval gate
eval/            build_labels.py · labels.json · harness.py
scripts/         benchmark.py · seeding · fine-tuning entry points
docs/            architecture · benchmark · eval_results · postmortem · training_results
```

MIT licensed. Built by [Akilan Manivannan](https://github.com/AkilanManivannanak).
