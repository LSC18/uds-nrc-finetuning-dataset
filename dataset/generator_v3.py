"""Leakage-reduced multi-profile dataset for closed-loop UDS request selection.

The model sees only the goal and observable request/response/NRC history. Ground-truth
ECU state and profile names are retained as evaluation metadata, never prompt features.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ecu import (
    ECU_PROFILES,
    SECURITY_ONLY_PROFILE,
    SESSION_ONLY_PROFILE,
    STRICT_PROFILE,
    EcuProfile,
    VirtualEcu,
)
from fuzzer.nrc_guided import format_hex, parse_response


TRAIN_PROFILES = (STRICT_PROFILE, SESSION_ONLY_PROFILE)
HELDOUT_PROFILE = SECURITY_ONLY_PROFILE


@dataclass(frozen=True)
class Step:
    step: int
    state_before: str
    request: str
    response: str
    nrc: str | None
    state_after: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    ecu_profile: str
    scenario: str
    goal: str
    initial_state: str
    seed: int
    collected_at: str
    steps: list[Step]
    result: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_for_index(index: int) -> EcuProfile:
    """Reserve every tenth episode for a behavior profile unseen during training."""
    if index % 10 == 0:
        return HELDOUT_PROFILE
    return TRAIN_PROFILES[index % len(TRAIN_PROFILES)]


def _unlock_sequence(profile: EcuProfile) -> list[bytes]:
    requests: list[bytes] = []
    if profile.security_requires_extended_session:
        requests.append(bytes.fromhex("10 03"))
    requests.extend((bytes.fromhex("27 01"), bytes.fromhex("27 02 BE EF")))
    return requests


def _write_recovery_sequence(profile: EcuProfile, write: bytes) -> list[bytes]:
    """Probe first, then recover solely from the profile's observable NRC behavior."""
    requests = [write]
    if profile.write_requires_extended_session:
        requests.append(bytes.fromhex("10 03"))
    if profile.write_requires_security:
        requests.extend((bytes.fromhex("27 01"), bytes.fromhex("27 02 BE EF")))
    requests.append(write)
    return requests


def _scenario_requests(
    rng: random.Random, profile: EcuProfile
) -> tuple[str, str, list[bytes], str]:
    value = rng.randrange(0x100)
    write = bytes((0x2E, 0xF1, 0xA0, value))
    scenario = rng.choices(
        (
            "session_success",
            "read_did_success",
            "write_probe_recovery",
            "security_probe_recovery",
            "nrc_invalid_key_recovery",
            "nrc_sequence_recovery",
            "nrc_range_recovery",
            "nrc_length_recovery",
            "reset_success",
            "failure",
        ),
        weights=(6, 7, 22, 12, 10, 10, 10, 10, 6, 7),
        k=1,
    )[0]

    if scenario == "session_success":
        target = rng.choice((0x01, 0x03))
        return scenario, f"0x{target:02X} 진단 세션 진입", [bytes((0x10, target))], "success"

    if scenario == "read_did_success":
        did = rng.choice((bytes.fromhex("F1 90"), bytes.fromhex("F1 A0")))
        return scenario, f"DID {format_hex(did)} 읽기", [bytes((0x22,)) + did], "success"

    if scenario == "write_probe_recovery":
        return (
            scenario,
            f"ECU 응답에 맞춰 DID F1 A0에 0x{value:02X} 쓰기",
            _write_recovery_sequence(profile, write),
            "success",
        )

    if scenario == "security_probe_recovery":
        requests = [bytes.fromhex("27 01")]
        if profile.security_requires_extended_session:
            requests.extend((bytes.fromhex("10 03"), bytes.fromhex("27 01")))
        requests.append(bytes.fromhex("27 02 BE EF"))
        return scenario, "ECU 응답에 맞춰 보안 잠금 해제", requests, "success"

    if scenario == "nrc_invalid_key_recovery":
        wrong_key = bytes((0x27, 0x02, rng.randrange(0x100), rng.randrange(0x100)))
        if wrong_key == bytes.fromhex("27 02 BE EF"):
            wrong_key = bytes.fromhex("27 02 00 00")
        requests = _unlock_sequence(profile)[:-1]
        requests.extend((wrong_key, bytes.fromhex("27 02 BE EF")))
        return scenario, "잘못된 키 NRC를 복구하고 보안 잠금 해제", requests, "success"

    if scenario == "nrc_sequence_recovery":
        requests: list[bytes] = []
        if profile.security_requires_extended_session:
            requests.append(bytes.fromhex("10 03"))
        requests.extend(
            (
                bytes.fromhex("27 02 BE EF"),
                bytes.fromhex("27 01"),
                bytes.fromhex("27 02 BE EF"),
            )
        )
        return scenario, "SecurityAccess 요청 순서를 복구하고 잠금 해제", requests, "success"

    if scenario == "nrc_range_recovery":
        bad_did = rng.randrange(0x10000)
        while bad_did in (VirtualEcu.VIN_DID, VirtualEcu.CONFIG_DID):
            bad_did = rng.randrange(0x10000)
        target_did = rng.choice((VirtualEcu.VIN_DID, VirtualEcu.CONFIG_DID))
        requests = [
            bytes((0x22,)) + bad_did.to_bytes(2, "big"),
            bytes((0x22,)) + target_did.to_bytes(2, "big"),
        ]
        return (
            scenario,
            f"지원되지 않는 DID 0x{bad_did:04X}를 복구해 0x{target_did:04X} 읽기",
            requests,
            "success",
        )

    if scenario == "nrc_length_recovery":
        variant = rng.choice(("session", "read", "tester"))
        junk = bytes(rng.randrange(0x100) for _ in range(rng.choice((0, 2, 3))))
        if variant == "session":
            requests = [bytes.fromhex("10") + junk, bytes.fromhex("10 03")]
            goal = "길이 NRC를 복구하고 extended_session 진입"
        elif variant == "read":
            requests = [bytes.fromhex("22 F1") + junk, bytes.fromhex("22 F1 90")]
            goal = "길이 NRC를 복구하고 VIN DID 읽기"
        else:
            requests = [bytes.fromhex("3E") + junk, bytes.fromhex("3E 00")]
            goal = "길이 NRC를 복구하고 TesterPresent 전송"
        return scenario, goal, requests, "success"

    if scenario == "reset_success":
        return (
            scenario,
            "확장 세션에서 ECU reset 후 초기 상태 복귀",
            [bytes.fromhex("10 03"), bytes.fromhex("11 01")],
            "success",
        )

    bad_request = rng.choice(
        (
            bytes((0x80 + rng.randrange(0x70), rng.randrange(0x100))),
            bytes((0x10,)),
            bytes((0x10, rng.choice((0x02, 0x7E, 0x7F)))),
            bytes((0x3E, rng.randrange(1, 0x80))),
        )
    )
    return scenario, f"잘못된 요청 {format_hex(bad_request)}의 실패 응답 기록", [bad_request], "failure"


def generate_episode(index: int, base_seed: int, collected_at: str) -> Episode:
    episode_seed = base_seed + index
    rng = random.Random(episode_seed)
    profile = profile_for_index(index)
    scenario, goal, requests, result = _scenario_requests(rng, profile)
    ecu = VirtualEcu(profile)
    steps: list[Step] = []

    for number, request in enumerate(requests, start=1):
        before = ecu.state_name()
        response = ecu.handle_request(request)
        parsed = parse_response(response)
        steps.append(
            Step(
                step=number,
                state_before=before,
                request=format_hex(request),
                response=format_hex(response),
                nrc=None if parsed.nrc is None else f"0x{int(parsed.nrc):02X}",
                state_after=ecu.state_name(),
            )
        )

    return Episode(
        episode_id=f"{profile.name}_run_{index:06d}",
        ecu_profile=profile.name,
        scenario=scenario,
        goal=goal,
        initial_state="default_session:locked",
        seed=episode_seed,
        collected_at=collected_at,
        steps=steps,
        result=result,
    )


def replay_episode(episode: Episode) -> bool:
    ecu = VirtualEcu(ECU_PROFILES[episode.ecu_profile])
    for step in episode.steps:
        if ecu.state_name() != step.state_before:
            return False
        response = ecu.handle_request(bytes.fromhex(step.request))
        if format_hex(response) != step.response:
            return False
        if ecu.state_name() != step.state_after:
            return False
    return True


def episode_to_samples(episode: Episode) -> Iterable[dict[str, object]]:
    history: list[dict[str, str | None]] = []
    for step in episode.steps:
        prompt = {"goal": episode.goal, "history": history}
        yield {
            "sample_id": f"{episode.episode_id}_step_{step.step}",
            "episode_id": episode.episode_id,
            "ecu_profile": episode.ecu_profile,
            "scenario": episode.scenario,
            "state_before": step.state_before,
            "state_after": step.state_after,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "너는 격리된 Virtual ECU 실험에서 목표와 관찰 가능한 UDS 이력만으로 "
                        "현재 상태를 추론해 다음 요청 바이트만 대문자 16진수로 선택한다."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
                },
                {"role": "assistant", "content": step.request},
            ],
        }
        history = [
            *history,
            {"request": step.request, "response": step.response, "nrc": step.nrc},
        ]


def _train_validation_split(episode_id: str, split_seed: int) -> str:
    digest = hashlib.sha256(f"{split_seed}:{episode_id}".encode()).digest()
    return "train" if int.from_bytes(digest[:8], "big") % 10 else "validation"


def split_for_episode(episode: Episode, split_seed: int) -> str:
    if episode.ecu_profile == HELDOUT_PROFILE.name:
        return "test"
    return _train_validation_split(episode.episode_id, split_seed)


class DatasetCollectorV3:
    def __init__(self, output_dir: Path, count: int, seed: int, interval: float = 0.0) -> None:
        if count < 1:
            raise ValueError("count must be at least 1")
        if interval < 0:
            raise ValueError("interval cannot be negative")
        self.output_dir = output_dir
        self.count = count
        self.seed = seed
        self.interval = interval

    def collect(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        episodes = [
            generate_episode(index, self.seed, started_at)
            for index in range(1, self.count + 1)
        ]
        with (self.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for index, episode in enumerate(episodes, start=1):
                stream.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
                if index == 1 or index % 1000 == 0 or index == self.count:
                    print(f"[COLLECT V3] {index}/{self.count}", flush=True)
                if self.interval:
                    time.sleep(self.interval)

        replay_failures = [e.episode_id for e in episodes if not replay_episode(e)]
        samples_by_split = {name: [] for name in ("train", "validation", "test")}
        raw_sample_count = 0
        for episode in episodes:
            split = split_for_episode(episode, self.seed)
            rows = list(episode_to_samples(episode))
            raw_sample_count += len(rows)
            samples_by_split[split].extend(rows)

        # Deduplicate globally so neither exact prompts nor labels leak across splits.
        signatures: set[str] = set()
        split_counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            unique_rows: list[dict[str, object]] = []
            for row in samples_by_split[split]:
                signature = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
                if signature in signatures:
                    continue
                signatures.add(signature)
                unique_rows.append(row)
            split_counts[split] = len(unique_rows)
            with (self.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row in unique_rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        nrc_counts = Counter(
            step.nrc for episode in episodes for step in episode.steps if step.nrc
        )
        state_counts = Counter(
            step.state_after for episode in episodes for step in episode.steps
        )
        profile_episode_counts = Counter(episode.ecu_profile for episode in episodes)
        sample_count = sum(split_counts.values())
        training_ready = (
            not replay_failures
            and split_counts["train"] >= 3000
            and split_counts["validation"] >= 300
            and split_counts["test"] >= 300
            and len(nrc_counts) >= 6
        )
        metadata = {
            "format_version": 2,
            "purpose": "closed_loop_state_inference_generalization",
            "training_ready": training_ready,
            "qlora_smoke_test_ready": training_ready,
            "prompt_excludes": ["current_state", "state_after", "ecu_profile"],
            "train_profiles": [profile.name for profile in TRAIN_PROFILES],
            "heldout_test_profile": HELDOUT_PROFILE.name,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed": self.seed,
            "episode_count": len(episodes),
            "profile_episode_counts": dict(sorted(profile_episode_counts.items())),
            "raw_sample_count": raw_sample_count,
            "sample_count": sample_count,
            "unique_sample_count": len(signatures),
            "exact_duplicate_count": raw_sample_count - sample_count,
            "exact_duplicate_ratio": round((raw_sample_count - sample_count) / raw_sample_count, 4),
            "split_sample_counts": split_counts,
            "scenario_counts": dict(sorted(Counter(e.scenario for e in episodes).items())),
            "nrc_counts": dict(sorted(nrc_counts.items())),
            "state_counts": dict(sorted(state_counts.items())),
            "replay_failures": replay_failures,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if replay_failures:
            raise RuntimeError(f"replay verification failed: {replay_failures[:5]}")
        print(f"[DONE V3] {sample_count} samples; replay failures=0", flush=True)
        return metadata
