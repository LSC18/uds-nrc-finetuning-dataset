#!/usr/bin/env python3
"""Validate JSONL shape, exact leakage, metadata counts, and UDS labels."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "full_v2"
SPLITS = ("train", "validation", "test")
HEX_REQUEST = re.compile(r"^[0-9A-F]{2}( [0-9A-F]{2})*$")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    signatures: dict[str, set[str]] = {}
    episode_ids: dict[str, set[str]] = {}
    counts: dict[str, int] = {}

    for split in SPLITS:
        rows = load_jsonl(DATA_DIR / f"{split}.jsonl")
        counts[split] = len(rows)
        signatures[split] = set()
        episode_ids[split] = set()
        for row in rows:
            messages = row["messages"]
            assert isinstance(messages, list) and len(messages) == 3
            assert [message["role"] for message in messages] == [
                "system",
                "user",
                "assistant",
            ]
            assert HEX_REQUEST.fullmatch(messages[-1]["content"])
            json.loads(messages[1]["content"])
            signatures[split].add(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            )
            episode_ids[split].add(row["episode_id"])
        assert len(signatures[split]) == len(rows)

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert signatures[left].isdisjoint(signatures[right])
        assert episode_ids[left].isdisjoint(episode_ids[right])

    assert counts == metadata["split_sample_counts"]
    assert sum(counts.values()) == metadata["sample_count"]
    assert metadata["replay_failures"] == []
    assert metadata["qlora_smoke_test_ready"] is True
    print(json.dumps({"status": "OK", "split_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
