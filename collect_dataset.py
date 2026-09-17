#!/usr/bin/env python3
"""Collect reproducible UDS episodes and QLoRA-ready JSONL samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import DatasetCollector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300, help="episode count")
    parser.add_argument("--seed", type=int, default=20260918, help="base RNG seed")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/pilot_v1"), help="output directory"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds to wait after each episode (for long-running collection)",
    )
    args = parser.parse_args()
    DatasetCollector(args.output_dir, args.count, args.seed, args.interval).collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
