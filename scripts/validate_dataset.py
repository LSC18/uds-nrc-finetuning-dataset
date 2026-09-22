#!/usr/bin/env python3
"""Validate JSONL shape, exact leakage, metadata counts, and UDS labels."""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
HEX_REQUEST = re.compile(r"^[0-9A-F]{2}( [0-9A-F]{2})*$")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "full_v3")
    args = parser.parse_args()
    data_dir = args.data_dir
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    signatures: dict[str, set[str]] = {}
    episode_ids: dict[str, set[str]] = {}
    counts: dict[str, int] = {}

    for split in SPLITS:
        rows = load_jsonl(data_dir / f"{split}.jsonl")
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
            prompt = json.loads(messages[1]["content"])
            if metadata.get("format_version", 1) >= 2:
                assert set(prompt) == {"goal", "history"}
                prompt_text = messages[1]["content"]
                assert "current_state" not in prompt_text
                assert "state_after" not in prompt_text
                assert "ecu_profile" not in prompt_text
                if split == "test":
                    assert row["ecu_profile"] == metadata["heldout_test_profile"]
                else:
                    assert row["ecu_profile"] in metadata["train_profiles"]
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
    print(
        json.dumps(
            {"status": "OK", "data_dir": str(data_dir), "split_counts": counts},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
