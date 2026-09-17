from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.generator import DatasetCollector, generate_episode, replay_episode


class DatasetTests(unittest.TestCase):
    def test_generation_is_deterministic_except_collection_time(self) -> None:
        first = generate_episode(1, 1234, "time-a").to_dict()
        second = generate_episode(1, 1234, "time-b").to_dict()
        first.pop("collected_at")
        second.pop("collected_at")
        self.assertEqual(first, second)

    def test_generated_episode_replays(self) -> None:
        for index in range(1, 50):
            self.assertTrue(replay_episode(generate_episode(index, 42, "test")))

    def test_collector_writes_disjoint_episode_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = DatasetCollector(output, count=100, seed=99).collect()
            split_episode_ids: dict[str, set[str]] = {}
            for split in ("train", "validation", "test"):
                records = [
                    json.loads(line)
                    for line in (output / f"{split}.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                split_episode_ids[split] = {r["episode_id"] for r in records}
            self.assertTrue(split_episode_ids["train"])
            self.assertTrue(split_episode_ids["validation"])
            self.assertTrue(split_episode_ids["test"])
            self.assertTrue(
                split_episode_ids["train"].isdisjoint(split_episode_ids["validation"])
            )
            self.assertTrue(
                split_episode_ids["train"].isdisjoint(split_episode_ids["test"])
            )
            self.assertEqual(metadata["replay_failures"], [])
            self.assertFalse(metadata["training_ready"])
            self.assertLessEqual(metadata["unique_sample_count"], metadata["sample_count"])


if __name__ == "__main__":
    unittest.main()
