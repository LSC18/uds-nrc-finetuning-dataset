# UDS NRC Next-Request Fine-tuning Dataset

격리된 Virtual ECU에서 현재 상태와 이전 UDS 요청·응답 이력을 보고 다음 UDS 요청을 선택하도록 학습하기 위한 합성 데이터셋이다.

실제 차량, 실제 ECU, CAN 인터페이스 또는 외부 네트워크와 통신하지 않는다.

## 데이터 현황

- Virtual ECU profile: `virtual_ecu_v2`
- 원본 episode: 7,200개
- 원본 step: 21,461개
- 정확 중복 제거 후 학습 샘플: 4,721개
- train / validation / test: 3,904 / 431 / 386
- split 간 정확 중복 및 episode 중복: 0건
- episode 재현 실패: 0건
- QLoRA smoke test 준비 상태: `true`

지원하는 UDS 동작:

- `0x10` DiagnosticSessionControl
- `0x11` ECUReset
- `0x22` ReadDataByIdentifier
- `0x27` SecurityAccess
- `0x2E` WriteDataByIdentifier
- `0x3E` TesterPresent
- NRC `0x11`, `0x12`, `0x13`, `0x24`, `0x31`, `0x33`, `0x35`, `0x7E`

## 파일

```text
full_v2/
├── episodes.jsonl     # 재현 가능한 원본 episode
├── train.jsonl        # QLoRA 학습 split
├── validation.jsonl   # 검증 split
├── test.jsonl         # 최종 평가 split
└── metadata.json      # 분포와 품질 검사 결과
```

상세 용도와 한계는 [`DATASET_CARD.md`](DATASET_CARD.md), 파일 무결성 값은 [`SHA256SUMS`](SHA256SUMS)에 기록한다.
Colab 실행 순서는 [`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md)에 정리했다.

학습 파일은 `messages` 기반 chat JSONL이다.

```json
{
  "sample_id": "run_000001_step_1",
  "episode_id": "run_000001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "10 03"}
  ]
}
```

## 검증

외부 패키지 없이 실행할 수 있다.

```bash
python3 scripts/validate_dataset.py
python3 -m unittest discover -s tests -v
shasum -a 256 -c SHA256SUMS
```

## 학습 직전 점검

Qwen tokenizer를 내려받은 뒤 전 데이터의 chat template 적용과 token 길이를 확인한다. 로컬처럼 CUDA가 없는 환경에서는 데이터 점검만 수행한다.

```bash
python3 scripts/preflight.py --skip-cuda
```

NVIDIA GPU 환경에서는 `--skip-cuda` 없이 실행하며 결과가 `status: ready`, `truncation_count: 0`인지 확인한다.

## QLoRA smoke test

NVIDIA GPU가 있는 Linux 또는 Colab 환경에서 실행한다. 기본 모델은 실행 시 명시하며, 모델 라이선스와 접근 권한은 별도로 확인해야 한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt

python3 scripts/preflight.py
python3 train_qlora.py --config configs/qlora_smoke.json
```

학습 데이터는 conversational prompt-completion 형식으로 변환되며 loss는 assistant completion에만 적용된다. 4-bit NF4와 `target_modules="all-linear"`를 사용한다.

학습 후 test exact-match 평가:

```bash
python3 evaluate_exact_match.py --adapter-path outputs/uds-nrc-smoke
```

학습 전 base model baseline도 같은 평가기로 기록한다.

```bash
python3 evaluate_exact_match.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output reports/baseline_predictions.jsonl
```

smoke test가 정상 종료되면 전체 설정으로 실행한다.

```bash
python3 train_qlora.py --config configs/qlora_full.json
python3 evaluate_exact_match.py --adapter-path outputs/uds-nrc-full
```

## 현재 준비 상태

- 데이터 schema 및 split 검증: 완료
- episode replay: 7,200개 모두 완료
- 정확 중복 및 split leakage 검사: 완료
- Qwen chat template 적용: 완료
- token 길이 검사: 최대 284, `max_length=512` 초과 0건
- smoke/full QLoRA 설정: 완료
- completion-only loss 구성: 완료
- 학습 후 exact-match 평가기: 완료
- 남은 작업: NVIDIA GPU에서 실제 weight update 실행

## 한계

이 데이터는 단일 합성 Virtual ECU의 결정론적 동작에서 생성됐다. `qlora_smoke_test_ready`는 학습 파이프라인을 시험할 수 있다는 뜻이며 실제 ECU에 일반화되는 연구용 최종 데이터라는 뜻은 아니다. 실제 연구에는 허가된 vCAN/ISO-TP 환경의 trace, 다양한 ECU profile, 시간 의존 상태, seed 다양화가 추가로 필요하다.
