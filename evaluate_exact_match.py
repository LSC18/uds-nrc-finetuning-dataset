#!/usr/bin/env python3
"""Evaluate a trained LoRA adapter on next-request exact match."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
HEX_REQUEST = re.compile(r"[0-9A-F]{2}(?: [0-9A-F]{2})*")


def normalize_prediction(text: str) -> str:
    match = HEX_REQUEST.search(text.upper().strip())
    return match.group(0) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter-path", type=Path)
    source.add_argument("--model-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/test_predictions.jsonl")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("evaluation requires an NVIDIA CUDA environment")
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_bf16 = torch.cuda.is_bf16_supported()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model_source = args.adapter_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model_class = AutoPeftModelForCausalLM if args.adapter_path else AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_source, quantization_config=quantization, device_map="auto"
    )
    model.eval()
    rows = [
        json.loads(line)
        for line in (ROOT / "full_v2/test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    correct = 0
    predictions: list[dict[str, object]] = []
    for row in rows:
        expected = row["messages"][-1]["content"]
        prompt = row["messages"][:-1]
        inputs = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        raw_prediction = tokenizer.decode(generated, skip_special_tokens=True)
        prediction = normalize_prediction(raw_prediction)
        matched = prediction == expected
        correct += int(matched)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "expected": expected,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "correct": matched,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    summary = {
        "model": str(model_source),
        "adapter": args.adapter_path is not None,
        "samples": len(rows),
        "correct": correct,
        "exact_match": correct / len(rows) if rows else 0.0,
        "predictions": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
