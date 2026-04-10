"""
DPO (Direct Preference Optimization) trainer.
Turns recruiter accept/reject decisions into preference pairs for RLHF.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_preference_dataset(
    feedback_records: list[dict],
    output_path: str | Path = "data/finetune/dpo_pairs.jsonl",
) -> int:
    """
    Convert ATS feedback into DPO training pairs.

    feedback_records: [{
        "prompt": "...",          # recruiter question / context
        "chosen": "...",          # answer recruiter accepted
        "rejected": "...",        # answer recruiter rejected / overrode
    }]

    Returns count of pairs written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = [r for r in feedback_records
             if r.get("prompt") and r.get("chosen") and r.get("rejected")]

    with open(output_path, "w", encoding="utf-8") as f:
        for record in valid:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(valid)} DPO preference pairs to {output_path}")
    return len(valid)


class DPOTrainer:
    """
    Fine-tune a model (or LoRA adapter) with DPO using recruiter preference pairs.

    Usage:
        trainer = DPOTrainer(
            model_name_or_path="models/talentra-lora/final_adapter",
            output_dir="models/talentra-dpo",
        )
        trainer.train("data/finetune/dpo_pairs.jsonl")
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/phi-3-mini-4k-instruct",
        output_dir: str = "models/talentra-dpo",
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 1,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs

    def _check_deps(self) -> bool:
        try:
            import trl, transformers, datasets  # noqa: F401
            return True
        except ImportError as e:
            logger.warning(f"DPO deps not available: {e}. Install: pip install trl transformers datasets")
            return False

    def train(self, dpo_data_path: str | Path) -> bool:
        """Run DPO fine-tuning. Returns True on success."""
        if not self._check_deps():
            return False

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer as TRLDPOTrainer
        from datasets import load_dataset

        logger.info(f"Loading model for DPO: {self.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, device_map=device_map, trust_remote_code=True
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, device_map=device_map, trust_remote_code=True
        )

        dataset = load_dataset("json", data_files={"train": str(dpo_data_path)})

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=self.learning_rate,
            report_to="none",
            remove_unused_columns=False,
        )

        dpo_trainer = TRLDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            beta=self.beta,
            train_dataset=dataset["train"],
            tokenizer=tokenizer,
        )

        logger.info("Starting DPO training...")
        dpo_trainer.train()
        dpo_trainer.save_model(str(self.output_dir / "final"))
        logger.info("DPO training complete.")
        return True
