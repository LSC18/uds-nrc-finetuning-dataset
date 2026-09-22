#!/usr/bin/env python3
"""Run every pre-training check that does not update model weights."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qlora_smoke.json")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/preflight.json")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = ROOT / config.get("data_dir", "full_v2")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not tokenizer.chat_template:
        raise SystemExit("FAIL: tokenizer has no chat template")

    token_stats: dict[str, dict[str, int]] = {}
    truncation_count = 0
    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in (data_dir / f"{split}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        lengths = [
            len(
                tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=False
                )["input_ids"]
            )
            for row in rows
        ]
        over_limit = sum(length > config["max_length"] for length in lengths)
        truncation_count += over_limit
        token_stats[split] = {
            "count": len(lengths),
            "min": min(lengths),
            "median": percentile(lengths, 0.5),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
            "over_max_length": over_limit,
        }

    cuda_available = torch.cuda.is_available()
    if not args.skip_cuda and not cuda_available:
        raise SystemExit("FAIL: CUDA is required; rerun with --skip-cuda for data-only checks")
    if truncation_count:
        raise SystemExit(f"FAIL: {truncation_count} samples exceed max_length")

    report = {
        "status": "ready" if cuda_available else "data_ready_waiting_for_cuda",
        "model_name": config["model_name"],
        "data_dir": str(data_dir),
        "max_length": config["max_length"],
        "truncation_count": truncation_count,
        "token_stats": token_stats,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": cuda_available,
            "bf16_supported": cuda_available and torch.cuda.is_bf16_supported(),
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
