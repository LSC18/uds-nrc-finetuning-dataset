#!/usr/bin/env python3
"""Collect leakage-reduced multi-profile UDS training data."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import DatasetCollectorV3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=15000, help="episode count")
    parser.add_argument("--seed", type=int, default=20260919, help="base RNG seed")
    parser.add_argument("--output-dir", type=Path, default=Path("full_v3"))
    parser.add_argument("--interval", type=float, default=0.0)
    args = parser.parse_args()
    DatasetCollectorV3(args.output_dir, args.count, args.seed, args.interval).collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
