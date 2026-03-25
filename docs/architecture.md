# Architecture sketch

```text
client/UI (Streamlit)
        |
        v
FastAPI serving layer
        |
        +--> role ingestion -> requirement extraction -> metadata store
        |
        +--> candidate ingestion -> parser -> chunker -> lexical index
        |
        +--> evaluation service -> retrieve evidence per requirement -> aggregate scores -> ATS dashboard
        |
        +--> copilot router -> evaluation-aware answer path OR targeted evidence search
        |
        +--> ops endpoints -> request metrics -> build metadata
```

## Data flow

1. **Ingest**: resume/JD files are parsed locally.
2. **Store**: metadata lands in `data/metadata.json`; content chunks land in `data/vectorstore/index.json`.
3. **Retrieve**: lexical search scores chunks with IDF-like weighting, phrase bonus, and contact-noise penalties.
4. **Infer**: ranking aggregates requirement-level evidence; copilot either uses evaluation results first or focused evidence retrieval.
5. **Feedback**: recruiter notes and ATS stage changes are persisted and shown on the dashboard.

## Trade-offs

- **Latency vs quality**: lexical retrieval is fast and cheap, but weaker than dense retrieval or cross-encoder reranking.
- **Freshness vs cost**: local indexing avoids API cost and quota failures, but does not provide semantic embeddings.
- **Reliability vs sophistication**: local rule-based extraction is deterministic and demo-safe, but less expressive than model-based extraction.
