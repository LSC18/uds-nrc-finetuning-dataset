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
```

## QLoRA smoke test

NVIDIA GPU가 있는 Linux 또는 Colab 환경에서 실행한다. 기본 모델은 실행 시 명시하며, 모델 라이선스와 접근 권한은 별도로 확인해야 한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt

python3 train_qlora.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/uds-nrc-smoke \
  --max-steps 50
```

본 학습 전에는 `--max-steps`를 늘리고 validation loss 및 `full_v2/test.jsonl`의 exact-match 정확도를 별도로 기록한다.

## 한계

이 데이터는 단일 합성 Virtual ECU의 결정론적 동작에서 생성됐다. `qlora_smoke_test_ready`는 학습 파이프라인을 시험할 수 있다는 뜻이며 실제 ECU에 일반화되는 연구용 최종 데이터라는 뜻은 아니다. 실제 연구에는 허가된 vCAN/ISO-TP 환경의 trace, 다양한 ECU profile, 시간 의존 상태, seed 다양화가 추가로 필요하다.
