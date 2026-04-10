"""
LoRA / QLoRA fine-tuning wrapper using HuggingFace PEFT + TRL.

Supports: Mistral-7B, Phi-3-mini, Llama-3, or any causal LM.
Falls back gracefully when GPU / transformers are unavailable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj"],
}

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "fp16": False,
    "bf16": False,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "optim": "adamw_torch",
}


class LoRATrainer:
    """
    Fine-tune a causal LM with LoRA adapters on Talentra SFT data.

    Usage:
        trainer = LoRATrainer(
            base_model="microsoft/phi-3-mini-4k-instruct",
            output_dir="models/talentra-lora",
        )
        trainer.train("data/finetune/train.jsonl")
        trainer.save()
    """

    def __init__(
        self,
        base_model: str = "microsoft/phi-3-mini-4k-instruct",
        output_dir: str = "models/talentra-lora",
        lora_config: dict | None = None,
        training_args: dict | None = None,
        use_4bit: bool = True,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.lora_config = {**DEFAULT_LORA_CONFIG, **(lora_config or {})}
        self.training_args = {**DEFAULT_TRAINING_ARGS, **(training_args or {})}
        self.use_4bit = use_4bit
        self._trainer = None
        self._model = None
        self._tokenizer = None

    def _check_deps(self) -> bool:
        try:
            import transformers, peft, trl, datasets  # noqa: F401
            return True
        except ImportError as e:
            logger.warning(f"Fine-tuning deps not available: {e}. Install: pip install transformers peft trl datasets bitsandbytes")
            return False

    def train(self, train_data_path: str | Path, eval_data_path: str | Path | None = None) -> bool:
        """
        Run LoRA fine-tuning.
        Returns True on success, False if deps are missing.
        """
        if not self._check_deps():
            return False

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset

        logger.info(f"Loading base model: {self.base_model}")

        # Quantization config
        bnb_config = None
        if self.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        if bnb_config:
            self._model = prepare_model_for_kbit_training(self._model)

        # Apply LoRA
        lora_cfg = LoraConfig(**self.lora_config)
        self._model = get_peft_model(self._model, lora_cfg)
        self._model.print_trainable_parameters()

        # Load dataset
        data_files = {"train": str(train_data_path)}
        if eval_data_path and Path(eval_data_path).exists():
            data_files["validation"] = str(eval_data_path)
        dataset = load_dataset("json", data_files=data_files)

        # Training arguments
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            report_to="none",
            **{k: v for k, v in self.training_args.items()
               if k not in ("fp16", "bf16") or torch.cuda.is_available()},
        )

        self._trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            args=args,
            max_seq_length=2048,
        )

        logger.info("Starting LoRA fine-tuning...")
        self._trainer.train()
        logger.info("Training complete.")
        return True

    def save(self, path: str | Path | None = None) -> Path:
        """Save LoRA adapter weights."""
        save_path = Path(path or self.output_dir / "final_adapter")
        save_path.mkdir(parents=True, exist_ok=True)
        if self._model:
            self._model.save_pretrained(str(save_path))
        if self._tokenizer:
            self._tokenizer.save_pretrained(str(save_path))
        logger.info(f"Adapter saved to {save_path}")
        return save_path

    def load_adapter(self, adapter_path: str | Path) -> Any:
        """Load a saved LoRA adapter for inference."""
        if not self._check_deps():
            return None
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base = AutoModelForCausalLM.from_pretrained(self.base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(adapter_path))
        return model.merge_and_unload(), tokenizer
