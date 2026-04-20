# LoRA Fine-tuning Results

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
| Eval gate | ✅ Passed — adapter promoted to active |

## Notes
- Trained on domain-specific hiring intelligence SFT data
- 4-bit quantization disabled (MPS; bitsandbytes not supported on Apple Silicon)
- models/ directory excluded from git; regenerate with `make finetune-lora`
