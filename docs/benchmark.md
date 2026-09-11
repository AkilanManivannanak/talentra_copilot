# Benchmark

_These are local latency measurements, not SLOs. An SLO is a commitment about production traffic under concurrency; this process serves one request at a time on one machine against a synthetic corpus._

Retrieval mode: `hybrid` · 12 candidates · 4 requirements · n=200 per measurement

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| evaluate cold | 9.943 | 12.498 | 29.153 |
| evaluate warm cache hit | 4.539 | 4.772 | 4.927 |
| copilot cold | 11.204 | 17.096 | 37.749 |
| copilot warm cache hit | 6.045 | 6.692 | 7.205 |
| candidate upload batch | 2229.841 | 2229.841 | 2229.841 |
| role create | 6.546 | 6.546 | 6.546 |

Cache speed-up on evaluate p95: **2.6x**. Warm numbers are cache hits, not request latency.

External API cost per request: **$0.000** (all models in-process, CPU).