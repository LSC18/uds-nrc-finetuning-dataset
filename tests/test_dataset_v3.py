from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.generator_v3 import (
    HELDOUT_PROFILE,
    DatasetCollectorV3,
    episode_to_samples,
    generate_episode,
    replay_episode,
)


class DatasetV3Tests(unittest.TestCase):
    def test_prompt_does_not_expose_profile_or_ground_truth_state(self) -> None:
        episode = generate_episode(10, 1000, "test")
        sample = next(iter(episode_to_samples(episode)))
        prompt = json.loads(sample["messages"][1]["content"])
        self.assertEqual(set(prompt), {"goal", "history"})
        prompt_text = sample["messages"][1]["content"]
        self.assertNotIn("current_state", prompt_text)
        self.assertNotIn("state_after", prompt_text)
        self.assertNotIn("ecu_profile", prompt_text)
        self.assertEqual(sample["ecu_profile"], HELDOUT_PROFILE.name)

    def test_all_profiles_replay(self) -> None:
        for index in range(1, 100):
            self.assertTrue(replay_episode(generate_episode(index, 500, "test")))

    def test_heldout_profile_is_test_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = DatasetCollectorV3(output, count=1000, seed=77).collect()
            profiles: dict[str, set[str]] = {}
            signatures: dict[str, set[str]] = {}
            for split in ("train", "validation", "test"):
                rows = [
                    json.loads(line)
                    for line in (output / f"{split}.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                profiles[split] = {row["ecu_profile"] for row in rows}
                signatures[split] = {
                    json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
                    for row in rows
                }
            self.assertNotIn(HELDOUT_PROFILE.name, profiles["train"])
            self.assertNotIn(HELDOUT_PROFILE.name, profiles["validation"])
            self.assertEqual(profiles["test"], {HELDOUT_PROFILE.name})
            self.assertTrue(signatures["train"].isdisjoint(signatures["validation"]))
            self.assertTrue(signatures["train"].isdisjoint(signatures["test"]))
            self.assertTrue(signatures["validation"].isdisjoint(signatures["test"]))
            self.assertEqual(metadata["replay_failures"], [])


if __name__ == "__main__":
    unittest.main()
