#!/usr/bin/env python3
"""
Run LoRA / QLoRA fine-tuning on a causal LM.

Usage:
    python scripts/finetune_lora.py --data data/finetune/train.jsonl
    python scripts/finetune_lora.py --data data/finetune/train.jsonl \
        --base-model microsoft/phi-3-mini-4k-instruct \
        --output models/talentra-lora \
        --epochs 3

Requires: pip install transformers peft trl datasets bitsandbytes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.finetuning.trainer import LoRATrainer
from app.finetuning.eval_gate import run_eval_gate


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Talentra Copilot")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument("--eval-data", default=None, help="Path to eval JSONL (optional)")
    parser.add_argument(
        "--base-model",
        default="microsoft/phi-3-mini-4k-instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--output", default="models/talentra-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--skip-gate", action="store_true", help="Skip eval gate after training")
    args = parser.parse_args()

    trainer = LoRATrainer(
        base_model=args.base_model,
        output_dir=args.output,
        training_args={"num_train_epochs": args.epochs},
        use_4bit=not args.no_4bit,
    )

    print(f"Starting LoRA fine-tuning on {args.base_model}...")
    success = trainer.train(
        train_data_path=args.data,
        eval_data_path=args.eval_data,
    )

    if not success:
        print(
            "Training could not run (missing dependencies or no GPU).\n"
            "Install: pip install transformers peft trl datasets bitsandbytes\n"
            "Training data generated successfully — you can run this on a GPU machine."
        )
        sys.exit(0)

    save_path = trainer.save()
    print(f"✓ Adapter saved to {save_path}")

    if not args.skip_gate:
        print("\nRunning eval gate...")
        gate = run_eval_gate(model_path=save_path, run_benchmark=True)
        if gate["passed"]:
            print("✓ Eval gate passed — model promoted to active.")
        else:
            print(f"✗ Eval gate failed: {gate['failures']}")
            print("Model saved but NOT promoted. Fix quality issues and re-run.")
            sys.exit(1)


if __name__ == "__main__":
    main()
