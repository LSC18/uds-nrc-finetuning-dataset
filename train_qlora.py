#!/usr/bin/env python3
"""CUDA QLoRA training entry point for UDS next-request prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "qlora_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = args.model_name or config["model_name"]
    output_dir = args.output_dir or ROOT / config["output_dir"]
    max_steps = args.max_steps if args.max_steps is not None else config["max_steps"]

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA requires an NVIDIA CUDA environment")

    import numpy as np
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.chat_template:
        raise SystemExit(f"model tokenizer has no chat template: {model_name}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_split(name: str) -> Dataset:
        rows = [
            json.loads(line)
            for line in (ROOT / "full_v2" / f"{name}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        return Dataset.from_list(rows)

    dataset = DatasetDict(
        {"train": load_split("train"), "validation": load_split("validation")}
    )

    def to_prompt_completion(example: dict[str, object]) -> dict[str, object]:
        messages = example["messages"]
        return {"prompt": messages[:-1], "completion": messages[-1:]}

    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset["train"].column_names,
        desc="Converting to completion-only training format",
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    trainer = SFTTrainer(
        model=model_name,
        processing_class=tokenizer,
        quantization_config=quantization,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            max_length=config["max_length"],
            max_steps=max_steps,
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type="cosine",
            logging_steps=config["logging_steps"],
            eval_strategy="steps",
            eval_steps=config["eval_steps"],
            save_strategy="steps",
            save_steps=config["save_steps"],
            save_total_limit=2,
            completion_only_loss=True,
            gradient_checkpointing=True,
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=seed,
            data_seed=seed,
            report_to="none",
        ),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {**train_result.metrics, **{f"final_{k}": v for k, v in eval_metrics.items()}}
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
