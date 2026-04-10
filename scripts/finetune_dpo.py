#!/usr/bin/env python3
"""
Run DPO training from recruiter preference pairs.

Usage:
    python scripts/finetune_dpo.py --model models/talentra-lora/final_adapter
    python scripts/finetune_dpo.py --pairs data/finetune/dpo_pairs.jsonl --beta 0.1

Requires: pip install trl transformers datasets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.finetuning.dpo import DPOTrainer, build_preference_dataset
from app.finetuning.eval_gate import run_eval_gate


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for Talentra Copilot")
    parser.add_argument(
        "--model",
        default="models/talentra-lora/final_adapter",
        help="Path to LoRA adapter or base model",
    )
    parser.add_argument("--pairs", default="data/finetune/dpo_pairs.jsonl", help="DPO pairs JSONL")
    parser.add_argument("--output", default="models/talentra-dpo", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta temperature")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--feedback-json",
        default=None,
        help="Raw ATS feedback JSON to convert to pairs first",
    )
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    # If raw feedback provided, build pairs first
    if args.feedback_json:
        feedback_path = Path(args.feedback_json)
        if feedback_path.exists():
            with open(feedback_path) as f:
                feedback = json.load(f)
            n = build_preference_dataset(feedback, output_path=args.pairs)
            print(f"Built {n} preference pairs → {args.pairs}")
        else:
            print(f"Feedback file not found: {args.feedback_json}")
            sys.exit(1)

    if not Path(args.pairs).exists():
        print(f"DPO pairs file not found: {args.pairs}")
        print("Run with --feedback-json to generate, or create pairs manually.")
        sys.exit(1)

    trainer = DPOTrainer(
        model_name_or_path=args.model,
        output_dir=args.output,
        beta=args.beta,
        num_train_epochs=args.epochs,
    )

    print(f"Starting DPO training from {args.model}...")
    success = trainer.train(args.pairs)

    if not success:
        print(
            "DPO training could not run (missing dependencies or no GPU).\n"
            "Install: pip install trl transformers datasets"
        )
        sys.exit(0)

    print(f"✓ DPO model saved to {args.output}/final")

    if not args.skip_gate:
        print("\nRunning eval gate...")
        gate = run_eval_gate(model_path=Path(args.output) / "final", run_benchmark=True)
        if gate["passed"]:
            print("✓ Eval gate passed — DPO model promoted.")
        else:
            print(f"✗ Eval gate failed: {gate['failures']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
