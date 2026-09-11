# Architecture

## Request paths

There are two entry points into the same engine. That is deliberate: the graph nodes are
constructed by `build_hiring_graph(services)` and close over the live `ServiceContainer`,
so the agentic path and the REST path cannot produce different rankings for the same input.
An earlier version had a parallel agent implementation that never ran; the lesson is that
two implementations of "rank these candidates" is one too many.

```text
REST                                        Agentic (LangGraph)
POST /roles/{id}/evaluate                   POST /roles/{id}/evaluate/agentic
POST /copilot/query                              │
      │                                          ▼
      │                                     screen ──► evaluate ──► audit_bias
      │                                                                │
      │                                                     route_question()
      │                                                       │         │
      ▼                                        copilot_qa ◄───┘         └──► evidence_search
RankingService / CopilotService  ◄───────────────┴────────────────────────────┘
                                                        │
                                                ⛔ interrupt_before
                                                   ats_update
                                                        │
                                    POST /agentic/runs/{run_id}/resume {approve}
                                                        ▼
                                                  ATS write applied
```

## Ingest

```text
upload → extract_text (pypdf / python-docx / utf-8)
       → clean_text          unicode, ligatures, PDF header noise
       → detect_sections     CONTACT / EXPERIENCE / SKILLS / PROJECTS / …
       → extract_skills      spaCy NER over a curated taxonomy, regex n-gram fallback
       → redact_pii          Presidio with an explicit entity allowlist, 7 regex fallbacks
       → parse_tenure        date ranges → years of experience
       → index REDACTED text
```

The indexed text is the **redacted** text when `REDACT_PII_ON_INGEST=true`. Contact details
are removed before they can ever be retrieved as evidence, rather than filtered at display
time. `DATE_TIME` is deliberately excluded from the redaction allowlist: tenure dates are
load-bearing evidence for the ranker.

## Retrieval

```text
                    ┌─────────────────────────────┐
query ─────────────►│ lexical: IDF over the        │──► ranked list A
                    │ filtered set, phrase and     │
                    │ bigram bonuses, contact-     │
                    │ noise penalty, meaningful-   │
                    │ overlap gate                 │
                    └─────────────────────────────┘
                    ┌─────────────────────────────┐
query ─────────────►│ dense: LSA (TF-IDF → SVD,    │──► ranked list B
                    │ corpus-fitted) or a          │
                    │ sentence-transformer         │
                    └─────────────────────────────┘
                                  │
                    reciprocal rank fusion, k=60
                          rrf(d) = Σ 1/(k + rank_r(d))
                                  │
                    optional cross-encoder rerank
                                  ▼
                            top-k evidence
```

RRF rather than weighted score interpolation: the two retrievers produce scores on
incomparable scales, so a weighted sum needs per-corpus calibration that silently rots.
RRF consumes only ranks.

**`search_grouped()` is the primary entry point.** One corpus-wide retrieval per
requirement, bucketed by candidate afterwards. Scoring each candidate with its own filtered
search made every hit rank 1, which collapsed RRF into a tie — see the v7→v8 postmortem.

## Storage

`data/metadata.json` — roles, candidates, documents, notes.
`data/vectorstore/index.json` — lexical chunks and tokens.
`data/vectorstore/dense_index.json` — vectors, tagged with the model that produced them so
a model change invalidates rather than silently mixes projections.

All writes go to a temp file in the same directory and land via `os.replace`, which is
atomic on POSIX and NTFS. Mutations hold a re-entrant lock. Lookups are dict-indexed by id
while the on-disk shape stays a readable list.

This is single-process. Caches are per-worker, so two uvicorn workers can serve different
cached answers. SQLite is the honest next step and is listed in the README's limitations.

## Observability

`GET /ops/metrics` — rolling window of request latencies, per-route p50/p95, error rate.
`GET /ops/build` — what this process can actually do. Dense backend, spaCy, Presidio and
LangGraph availability are **probed at runtime**, not asserted, so a deployment answers the
question rather than the documentation.
`GET /metrics` — Prometheus scrape.
`X-Request-ID` on every response.

## Degradation

Every optional component falls back, and the active configuration is always visible at
`/ops/build`:

| Missing | Behaviour |
|---|---|
| scikit-learn | dense unavailable; `hybrid` reports itself as `lexical` |
| sentence-transformers | LSA still provides dense retrieval |
| spaCy model | regex taxonomy for skills |
| Presidio | 7 regex PII patterns |
| langgraph | sequential executor honouring the same interrupt contract |
