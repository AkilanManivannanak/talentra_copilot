#!/usr/bin/env python3
"""
Generate fine-tuning training data from eval fixtures and optional ATS feedback.

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --seed eval/demo_seed.json --out data/finetune/train.jsonl
    python scripts/generate_training_data.py --feedback data/ats_feedback.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure app is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.finetuning.data_generator import generate_training_data, export_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate Talentra fine-tuning data")
    parser.add_argument("--seed", default="eval/demo_seed.json", help="Path to demo_seed.json")
    parser.add_argument("--feedback", default=None, help="Path to ATS feedback JSON (optional)")
    parser.add_argument("--out", default="data/finetune/train.jsonl", help="Output JSONL path")
    parser.add_argument("--preview", action="store_true", help="Print first 2 examples and exit")
    args = parser.parse_args()

    feedback_records = []
    if args.feedback:
        feedback_path = Path(args.feedback)
        if feedback_path.exists():
            with open(feedback_path) as f:
                feedback_records = json.load(f)
            print(f"Loaded {len(feedback_records)} feedback records from {args.feedback}")
        else:
            print(f"Warning: feedback file not found: {args.feedback}")

    examples = generate_training_data(
        seed_path=args.seed,
        feedback_records=feedback_records or None,
    )

    if not examples:
        print("No training examples generated. Check seed file path.")
        sys.exit(1)

    if args.preview:
        for ex in examples[:2]:
            print(json.dumps(ex, indent=2))
        print(f"\n... {len(examples)} total examples")
        return

    count = export_jsonl(examples, args.out)
    print(f"✓ Wrote {count} training examples to {args.out}")
    print(f"  Next: python scripts/finetune_lora.py --data {args.out}")


if __name__ == "__main__":
    main()
