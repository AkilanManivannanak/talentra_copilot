# Postmortem: what broke and how this version fixed it

## Incident 1: OpenAI quota and auth failures blocked ingestion
- **Symptom**: resume upload returned 500 because embeddings failed.
- **Root cause**: the earlier version depended on OpenAI embeddings and chat completions, and the API project had invalid or exhausted quota.
- **Fix**: replaced the default path with local lexical retrieval and rule-based extraction; external API spend is now optional and disabled by default.

## Incident 2: brittle PDF ingestion caused whole-batch failure
- **Symptom**: one malformed PDF crashed the entire upload request.
- **Root cause**: upload processing was all-or-nothing with no per-file error isolation.
- **Fix**: upload now supports partial success; unreadable files return structured failures instead of raw 500s.

## Incident 3: ghost candidates after failed ingest
- **Symptom**: failed uploads still left candidate records behind.
- **Root cause**: metadata was written before ingest fully succeeded.
- **Fix**: candidate/document cleanup now runs on failure and orphan candidates are removed.

## Incident 4: Copilot contradicted evaluation ranking
- **Symptom**: Evaluate ranked Akila first while Copilot sometimes answered Jaxon or Esha.
- **Root cause**: comparison questions were answered from raw lexical hits instead of the evaluation results.
- **Fix**: ranking/comparison questions now route to evaluation results first; targeted skill questions use evidence search separately.

## Incident 5: header/contact junk polluted evidence
- **Symptom**: citations surfaced email, LinkedIn, and phone blocks.
- **Root cause**: chunks from resume headers were indexed without downweighting.
- **Fix**: low-signal chunk filtering, contact-noise penalties, and evidence deduping were added.

---

## v6 → v7: the integration that never landed

**Symptom.** The README described LangGraph orchestration, five agents, spaCy NER, Presidio
redaction, a Chroma→FAISS→lexical vectorstore cascade, and LoRA promotion. The running
application did none of it.

**Root cause.** Commit `377037e` wiped the repo for a rebuild. Commit `bbee4ab` added the
entire v2 layer — `app/preprocessing/`, `app/langchain_layer/`, `app/graph/`, `app/agents/`,
`app/finetuning/` — in a single commit, authored *alongside* the working `app/services/`
stack rather than replacing it. The commit that would have wired them together never landed.
The result:

- `app/graph/hiring_graph.py` was imported by nothing.
- `app/routers/evaluation.py` was the only importer of the agents, and was never registered
  in `main.py`. It also read `Settings.metadata_path` and `Settings.vectorstore_dir`, neither
  of which exists, and declared a route path that collided with `roles.py`.
- `langgraph`, `langchain`, `spacy`, `presidio`, `chromadb`, `faiss` and `peft` were absent
  from `requirements.txt`, so even the import guards resolved to fallbacks.
- `_get_embeddings()` in the langchain-layer vectorstore returned `None` unconditionally,
  making the advertised Chroma and FAISS branches unreachable by construction.

**Why nobody noticed.** Three commits after the v2 layer landed, `9a5d2b4` ("robust CI —
remove pytest dependency") removed `pytest` from CI. The remaining jobs ran `py_compile`
over a hand-maintained file list that named four dead modules and omitted `app/services/`
entirely. So CI was syntax-checking code that does not execute while ignoring the code that
does, and the test suite went red on `main` without turning the badge red.

**What that hid.** `tests/test_ops.py` had been failing the whole time. `/ops/metrics`
returned `{"metrics": {}}` and `/ops/build` returned `{"build_info": {}}`: both response
models declared a single dict field and pydantic v2's `extra="ignore"` silently dropped
every key passed to them. The observability screen in the demo was rendering empty payloads.

**Fixes.** Registered a real agentic route bound to the live services; pinned the
dependencies that were being imported optionally; deleted the four duplicate `core/` modules
and the second vectorstore implementation; gave the ops schemas their real shape; put
`pytest` back as the first CI gate with `ruff` ahead of it.

**Lesson.** A `try/ImportError` fallback around a headline feature is indistinguishable from
that feature not existing, and it is *designed* to be silent. Every optional capability is
now probed at runtime and reported at `/ops/build`, so the deployment answers the question
rather than the README.

---

## v7 → v8: rank fusion that made ranking worse

**Symptom.** The first run of the new retrieval ablation showed hybrid scoring **below both
of its own inputs**: nDCG@5 of 0.728 against 0.780 lexical and 0.799 dense. Fusing two
retrievers should not be worse than either.

**Root cause.** `RankingService` scored each candidate with its own retrieval call filtered
to `entity_id`. With one document in the eligible set, every hit is rank 1 — so reciprocal
rank fusion assigned every candidate an identical fused score and the ranking collapsed into
a tie. The bug was invisible in lexical mode, which uses raw scores rather than ranks, which
is why it survived until the ablation existed to expose it.

**Fix.** `HybridRetriever.search_grouped()`: one corpus-wide retrieval per requirement,
bucketed by candidate afterwards. Ranks recover their meaning, IDF becomes global rather
than per-document, and the search count drops from `N_candidates × N_requirements` to
`N_requirements`. The screener was changed the same way and now judges candidates relative
to the pool instead of against an absolute threshold.

**Lesson.** The ablation paid for itself on its first run. A fusion method that consumes
ranks needs a candidate set large enough for ranks to carry information — and the only
reason this was caught is that there was now a metric capable of failing.
