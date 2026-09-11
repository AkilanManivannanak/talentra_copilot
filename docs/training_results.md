# LoRA fine-tuning results

## Run: Qwen2.5-0.5B-Instruct — April 2026

| Metric | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Trainable params | 1,081,344 / 495,114,112 (0.22%) |
| LoRA config | r=16, alpha=32, dropout=0.05, target=[q_proj, v_proj] |
| Epochs | 3 |
| Training examples | 12 |
| Total tokens | 6,117 |
| Train loss | 4.079 |
| Mean token accuracy | 0.3826 |
| Training runtime | 32s (Apple M4, MPS backend) |

## What these numbers mean

A train loss of 4.079 and mean token accuracy of 0.38 on a 0.5B model after three epochs
over 6,117 tokens means **the adapter learned essentially nothing**. Twelve examples is far
below what a LoRA run needs to move a model. The run demonstrates that the pipeline
executes end to end; it does not demonstrate a model improvement, and this file previously
described the adapter as "promoted to active" on the strength of it.

## Correction to the earlier claim

The earlier version of this document recorded "Eval gate ✅ Passed — adapter promoted to
active". That was not true in any meaningful sense:

- The gate ran `scripts/benchmark.py`, which builds the FastAPI app and exercises the
  **rule-based lexical pipeline**. The adapter at `model_path` was never loaded, never
  inferenced, and never compared against anything.
- On "pass" it wrote `models/active_model.json`, a pointer file that **nothing in the
  repository reads**.

So an adapter could be "promoted" without a single token passing through it.

## What the gate does now

`app/finetuning/eval_gate.py` requires a model in the loop:

1. Base model and adapter answer the same held-out prompts, built from the evaluation
   corpus and distinct from the SFT generation source.
2. Each answer is scored for groundedness with the same `FaithfulnessScorer` the serving
   path uses.
3. Promotion requires an absolute groundedness of ≥ 0.60 **and** a gain of ≥ 0.05 over the
   base model, plus a passing serving-path latency gate.

If `transformers`/`peft` are unavailable or no adapter path is supplied, the gate **refuses
to promote** and says so. It does not pass on absence of evidence.

## What is still not true

`SummaryService` is rule-based synthesis over retrieved evidence. It loads no causal LM.
`models/active_model.json` is an offline record of the gate's outcome and is not consumed
by the request path — the file itself now says so. Wiring a generation backend into
`SummaryService` is the work that would make any of this load-bearing.

## Reproducing

```bash
pip install -r requirements-ml.txt
python scripts/generate_training_data.py --seed eval/labels.json --out data/finetune/train.jsonl
python scripts/finetune_lora.py --data data/finetune/train.jsonl --output models/talentra-lora
```

The `models/` directory is git-ignored.
