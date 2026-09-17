#!/usr/bin/env python3
"""Minimal CUDA QLoRA smoke-training entry point for the UDS chat dataset."""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="outputs/uds-nrc-smoke")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA smoke test requires an NVIDIA CUDA environment")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization,
        device_map="auto",
    )
    dataset = load_dataset(
        "json",
        data_files={
            "train": "full_v2/train.jsonl",
            "validation": "full_v2/validation.jsonl",
        },
    )

    def render(example: dict[str, object]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(render, remove_columns=dataset["train"].column_names)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=args.max_length,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_steps=25,
            bf16=True,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
