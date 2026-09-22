# Dataset Card: UDS NRC Next-Request v3

## Summary

목표와 관찰 가능한 UDS 요청·응답·NRC 이력만으로 다음 UDS 요청 바이트를 예측하는 합성 supervised fine-tuning 데이터셋이다. 실제 ECU 내부 상태와 profile 식별자는 학습 프롬프트에서 제외한다.

- Language: Korean instructions, hexadecimal UDS requests and responses
- Format: conversational JSONL with evaluation metadata
- Source: deterministic isolated Virtual ECU profiles
- Personal or vehicle data: none
- Real vehicle traces: none

## Splits

| Split | Samples | ECU profiles |
|---|---:|---|
| train | 8,266 | strict, session-only |
| validation | 777 | strict, session-only |
| test | 1,364 | security-only held-out |
| total | 10,407 | 3 synthetic profiles |

동일 `messages`는 split 사이에 겹치지 않는다. test는 학습 중 등장하지 않은 ECU 동작 profile만 포함한다.

## Generation

- Base seed: `20260919`
- Raw episodes: 15,000
- Raw steps: 41,980
- Exact duplicates removed: 31,573
- Replay failures: 0
- Prompt-excluded fields: `current_state`, `state_after`, `ecu_profile`

## Token Lengths

`Qwen/Qwen2.5-1.5B-Instruct` tokenizer 기준이다.

| Split | Median | P95 | P99 | Max | Over 512 |
|---|---:|---:|---:|---:|---:|
| train | 138 | 196 | 228 | 228 | 0 |
| validation | 135 | 196 | 196 | 228 | 0 |
| test | 138 | 202 | 202 | 202 | 0 |

## Intended Use

- NRC 피드백 기반 다음 요청 선택 QLoRA
- 1.5B와 7B 모델 크기 비교
- 학습에 없던 합성 ECU profile 평가
- scenario별 실패 분석

## Out-of-Scope Use

- 실제 차량 또는 실제 ECU에 대한 안전성 판단
- 제조사별 진단 로직 일반화 주장
- 허가받지 않은 차량·ECU 테스트
- 실제 공격 성공 가능성 평가

## Limitations

세 profile 모두 같은 코드 기반의 합성 Virtual ECU다. 실제 ISO-TP 전송, timing, P2/P2*, session timeout, 제조사별 DID와 서비스 차이, 다양한 seed-key 알고리즘을 포함하지 않는다. held-out profile 평가는 코드 경로가 다른 실제 ECU에 대한 일반화를 의미하지 않는다.

## Integrity

```bash
python3 scripts/build_checksums.py
shasum -a 256 -c SHA256SUMS
```
