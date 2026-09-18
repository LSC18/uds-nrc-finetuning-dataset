# Dataset Card: UDS NRC Next-Request v2

## Summary

현재 Virtual ECU 상태, 목표, 이전 UDS 요청·응답 이력을 입력으로 받아 다음 UDS 요청 바이트를 예측하는 합성 supervised fine-tuning 데이터셋이다.

- Language: Korean instructions, hexadecimal UDS requests and responses
- Format: conversational JSONL
- License: repository license not specified
- Source: deterministic local Virtual ECU only
- Personal or vehicle data: none
- Real vehicle traces: none

## Splits

| Split | Samples |
|---|---:|
| train | 3,904 |
| validation | 431 |
| test | 386 |
| total | 4,721 |

동일 `messages` 샘플과 동일 episode가 split 사이에 겹치지 않는다.

## Generation

- Base seed: `20260918`
- ECU profile: `virtual_ecu_v2`
- Raw episodes: 7,200
- Raw steps: 21,461
- Exact duplicates removed: 16,740
- Replay failures: 0

## Token Lengths

`Qwen/Qwen2.5-1.5B-Instruct` tokenizer와 chat template 기준이다.

| Split | Median | P95 | P99 | Max | Over 512 |
|---|---:|---:|---:|---:|---:|
| train | 187 | 278 | 284 | 284 | 0 |
| validation | 188 | 239 | 284 | 284 | 0 |
| test | 158 | 242 | 284 | 284 | 0 |

## Intended Use

- QLoRA 코드와 데이터 로딩 파이프라인 smoke test
- 다음 UDS 요청 선택 모델의 초기 실험
- NRC 및 ECU 상태별 오류 분석 코드 개발

## Out-of-Scope Use

- 실제 차량 또는 실제 ECU에 대한 안전성 판단
- 제조사별 진단 로직 일반화 주장
- 허가받지 않은 차량·ECU 테스트
- 실제 공격 성공 가능성 평가

## Limitations

단일 결정론적 Virtual ECU에서 생성한 합성 데이터다. 실제 ISO-TP 전송, timing, P2/P2*, session timeout, seed-key 알고리즘 다양성, 제조사별 DID와 서비스 동작을 포함하지 않는다. 따라서 현재 데이터의 `qlora_smoke_test_ready=true`는 학습 파이프라인 실행 준비를 의미하며 연구용 최종 데이터 완성을 의미하지 않는다.

## Integrity

`SHA256SUMS`로 versioned JSONL과 metadata의 SHA-256을 검증할 수 있다.

```bash
shasum -a 256 -c SHA256SUMS
```
