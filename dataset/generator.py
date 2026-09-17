"""Deterministic episode collection and next-request sample conversion."""

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

from ecu import VirtualEcu
from fuzzer.nrc_guided import format_hex, parse_response


ECU_PROFILE = "virtual_ecu_v2"


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


def _scenario_requests(rng: random.Random) -> tuple[str, str, list[bytes], str]:
    """Return a goal-directed request sequence with optional NRC recovery."""
    value = rng.randrange(0x100)
    write = bytes((0x2E, 0xF1, 0xA0, value))
    unlock = [bytes.fromhex("10 03"), bytes.fromhex("27 01"), bytes.fromhex("27 02 BE EF")]
    scenario = rng.choices(
        (
            "session_success",
            "read_did_success",
            "security_unlock_success",
            "write_config_success",
            "nrc_session_recovery",
            "nrc_security_recovery",
            "nrc_invalid_key_recovery",
            "nrc_sequence_recovery",
            "nrc_range_recovery",
            "nrc_length_recovery",
            "reset_success",
            "failure",
        ),
        weights=(8, 9, 9, 14, 9, 11, 8, 8, 8, 8, 4, 4),
        k=1,
    )[0]

    if scenario == "session_success":
        target = rng.choice((0x01, 0x03))
        goal = f"0x{target:02X} 진단 세션 진입"
        requests = [bytes((0x10, target))]
        result = "success"
    elif scenario == "read_did_success":
        did = rng.choice((bytes.fromhex("F1 90"), bytes.fromhex("F1 A0")))
        goal = f"DID {format_hex(did)} 읽기"
        requests = [bytes((0x22,)) + did]
        result = "success"
    elif scenario == "security_unlock_success":
        goal = "extended_session에서 보안 잠금 해제"
        requests = unlock
        result = "success"
    elif scenario == "write_config_success":
        goal = f"DID F1 A0 값을 0x{value:02X}로 쓰기"
        requests = [*unlock, write]
        result = "success"
    elif scenario == "nrc_session_recovery":
        goal = "세션 NRC를 복구하고 보안 잠금 해제"
        requests = [bytes.fromhex("27 01"), *unlock]
        result = "success"
    elif scenario == "nrc_security_recovery":
        goal = f"보안 NRC를 복구하고 DID F1 A0에 0x{value:02X} 쓰기"
        requests = [bytes.fromhex("10 03"), write, *unlock[1:], write]
        result = "success"
    elif scenario == "nrc_invalid_key_recovery":
        wrong_key = bytes((0x27, 0x02, rng.randrange(0x100), rng.randrange(0x100)))
        if wrong_key == bytes.fromhex("27 02 BE EF"):
            wrong_key = bytes.fromhex("27 02 00 00")
        goal = "잘못된 키 NRC를 복구하고 보안 잠금 해제"
        requests = [*unlock[:2], wrong_key, unlock[2]]
        result = "success"
    elif scenario == "nrc_sequence_recovery":
        goal = "SecurityAccess 순서 NRC를 복구하고 잠금 해제"
        requests = [unlock[0], unlock[2], *unlock[1:]]
        result = "success"
    elif scenario == "nrc_range_recovery":
        bad_did = rng.randrange(0x10000)
        while bad_did in (VirtualEcu.VIN_DID, VirtualEcu.CONFIG_DID):
            bad_did = rng.randrange(0x10000)
        target_did = rng.choice((VirtualEcu.VIN_DID, VirtualEcu.CONFIG_DID))
        goal = f"지원되지 않는 DID NRC를 복구하고 0x{target_did:04X} 읽기"
        requests = [
            bytes((0x22,)) + bad_did.to_bytes(2, "big"),
            bytes((0x22,)) + target_did.to_bytes(2, "big"),
        ]
        result = "success"
    elif scenario == "nrc_length_recovery":
        choice = rng.choice(("session", "read", "tester"))
        if choice == "session":
            goal = "길이 NRC를 복구하고 extended_session 진입"
            requests = [bytes.fromhex("10"), bytes.fromhex("10 03")]
        elif choice == "read":
            goal = "길이 NRC를 복구하고 VIN DID 읽기"
            requests = [bytes.fromhex("22 F1"), bytes.fromhex("22 F1 90")]
        else:
            goal = "길이 NRC를 복구하고 TesterPresent 전송"
            requests = [bytes.fromhex("3E"), bytes.fromhex("3E 00")]
        result = "success"
    elif scenario == "reset_success":
        goal = "확장 세션에서 ECU reset 후 기본 잠금 상태 복귀"
        requests = [bytes.fromhex("10 03"), bytes.fromhex("11 01")]
        result = "success"
    else:
        goal = "잘못된 요청의 실패 응답 기록"
        requests = [
            rng.choice(
                (
                    bytes((0x99, rng.randrange(0x100))),
                    bytes((0x10,)),
                    bytes((0x10, 0x7E)),
                    bytes((0x3E, 0x01)),
                )
            )
        ]
        result = "failure"
    return scenario, goal, requests, result


def generate_episode(index: int, base_seed: int, collected_at: str) -> Episode:
    episode_seed = base_seed + index
    rng = random.Random(episode_seed)
    scenario, goal, requests, result = _scenario_requests(rng)
    ecu = VirtualEcu()
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
        episode_id=f"run_{index:06d}",
        ecu_profile=ECU_PROFILE,
        scenario=scenario,
        goal=goal,
        initial_state="default_session:locked",
        seed=episode_seed,
        collected_at=collected_at,
        steps=steps,
        result=result,
    )


def replay_episode(episode: Episode) -> bool:
    """Replay recorded requests and compare all observable ECU outputs/states."""
    ecu = VirtualEcu()
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
        prompt = {
            "ecu_profile": episode.ecu_profile,
            "goal": episode.goal,
            "current_state": step.state_before,
            "history": history,
        }
        yield {
            "sample_id": f"{episode.episode_id}_step_{step.step}",
            "episode_id": episode.episode_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "너는 격리된 Virtual ECU 실험에서 현재 상태와 UDS 이력을 보고 "
                        "다음 요청 바이트만 대문자 16진수로 선택한다."
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
            {
                "request": step.request,
                "response": step.response,
                "nrc": step.nrc,
                "state_after": step.state_after,
            },
        ]


def _split_for_episode(episode_id: str, split_seed: int) -> str:
    digest = hashlib.sha256(f"{split_seed}:{episode_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


class DatasetCollector:
    def __init__(
        self,
        output_dir: Path,
        count: int,
        seed: int,
        interval: float = 0.0,
    ) -> None:
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
        episodes_path = self.output_dir / "episodes.jsonl"
        split_paths = {
            name: self.output_dir / f"{name}.jsonl"
            for name in ("train", "validation", "test")
        }
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        episodes: list[Episode] = []

        with episodes_path.open("w", encoding="utf-8") as stream:
            for index in range(1, self.count + 1):
                episode = generate_episode(index, self.seed, started_at)
                stream.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
                stream.flush()
                episodes.append(episode)
                if index == 1 or index % 25 == 0 or index == self.count:
                    print(f"[COLLECT] {index}/{self.count}", flush=True)
                if self.interval:
                    time.sleep(self.interval)

        replay_failures = [e.episode_id for e in episodes if not replay_episode(e)]
        split_counts = {name: 0 for name in split_paths}
        raw_sample_count = 0
        sample_count = 0
        sample_signatures: set[str] = set()
        streams = {
            name: path.open("w", encoding="utf-8") for name, path in split_paths.items()
        }
        try:
            for episode in episodes:
                split = _split_for_episode(episode.episode_id, self.seed)
                for sample in episode_to_samples(episode):
                    raw_sample_count += 1
                    signature = json.dumps(
                        sample["messages"], ensure_ascii=False, sort_keys=True
                    )
                    if signature in sample_signatures:
                        continue
                    sample_signatures.add(signature)
                    streams[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                    split_counts[split] += 1
                    sample_count += 1
        finally:
            for stream in streams.values():
                stream.close()

        nrc_counts = Counter(
            step.nrc for episode in episodes for step in episode.steps if step.nrc
        )
        state_counts = Counter(
            step.state_after for episode in episodes for step in episode.steps
        )
        smoke_test_ready = (
            not replay_failures
            and sample_count >= 500
            and min(split_counts.values()) >= 50
            and len(nrc_counts) >= 6
        )
        metadata = {
            "format_version": 1,
            "purpose": "schema_pipeline_pilot",
            "training_ready": False,
            "qlora_smoke_test_ready": smoke_test_ready,
            "ecu_profile": ECU_PROFILE,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed": self.seed,
            "episode_count": len(episodes),
            "raw_sample_count": raw_sample_count,
            "sample_count": sample_count,
            "unique_sample_count": len(sample_signatures),
            "exact_duplicate_count": raw_sample_count - sample_count,
            "exact_duplicate_ratio": round(
                (raw_sample_count - sample_count) / raw_sample_count, 4
            ),
            "split_sample_counts": split_counts,
            "scenario_counts": {
                name: sum(e.scenario == name for e in episodes)
                for name in sorted({e.scenario for e in episodes})
            },
            "nrc_counts": dict(sorted(nrc_counts.items())),
            "state_counts": dict(sorted(state_counts.items())),
            "replay_failures": replay_failures,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if replay_failures:
            raise RuntimeError(f"replay verification failed: {replay_failures[:5]}")
        print(f"[DONE] {sample_count} samples; replay failures=0", flush=True)
        return metadata
