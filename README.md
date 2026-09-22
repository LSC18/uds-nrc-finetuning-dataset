# UDS NRC Closed-Loop Fine-tuning Dataset

격리된 Virtual ECU에서 목표와 관찰 가능한 UDS 요청·응답·NRC 이력만 보고 다음 요청을 선택하는 모델을 학습하기 위한 합성 데이터셋이다. 실제 차량, 실제 ECU, CAN 인터페이스 또는 외부 네트워크와 통신하지 않는다.

## v3 데이터 현황

- 원본 episode: 15,000개
- 원본 step: 41,980개
- 정확 중복 제거 후 고유 샘플: 10,407개
- train / validation / held-out test: 8,266 / 777 / 1,364
- 학습 ECU profile: `strict_session_security`, `session_only`
- 미관측 테스트 profile: `security_only_heldout`
- 프롬프트 제외 항목: `current_state`, `state_after`, `ecu_profile`
- split 간 정확 중복: 0건
- episode 재현 실패: 0건
- QLoRA 준비 상태: `true`

지원 동작은 `0x10`, `0x11`, `0x22`, `0x27`, `0x2E`, `0x3E`이며 NRC `0x11`, `0x12`, `0x13`, `0x24`, `0x31`, `0x33`, `0x35`, `0x7E`를 생성한다.

## 파일

```text
full_v3/
├── episodes.jsonl
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── metadata.json
```

`full_v2/`는 상태 정답을 입력으로 제공하던 이전 파이프라인 검증용 데이터로 보존한다. 논문 실험에는 `full_v3/`를 사용한다.

학습 샘플의 `messages`에는 목표와 관찰 이력만 포함된다. `ecu_profile`, `state_before`, `state_after`는 평가·오류 분석용 row metadata이며 모델 입력에 포함되지 않는다.

## 생성 및 검증

```bash
python3 collect_dataset_v3.py --count 15000 --output-dir full_v3
python3 scripts/validate_dataset.py --data-dir full_v3
python3 -m unittest discover -s tests -v
python3 scripts/build_checksums.py
shasum -a 256 -c SHA256SUMS
```

## 학습 직전 점검

[![Open v3 smoke test in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSC18/uds-nrc-finetuning-dataset/blob/main/notebooks/qlora_v3_colab.ipynb)

Colab에서는 위 노트북을 열고 위에서부터 순서대로 실행한다. Colab이 제공하는
CUDA 호환 PyTorch를 그대로 사용하며 `requirements-colab.txt`는 PyTorch를
업그레이드하지 않는다. 기존 설치 셀로 실행해서 `torchvision::nms` 오류가 난
런타임은 **런타임 → 세션 다시 시작 및 모두 실행**으로 초기화한다.

```bash
python3 scripts/preflight.py \
  --config configs/qlora_v3_smoke_1.5b.json \
  --skip-cuda \
  --report reports/preflight_v3_1.5b.json
```

NVIDIA GPU에서는 `--skip-cuda`를 제거한다. 현재 최대 길이는 228토큰이며 `max_length=512` 초과 샘플은 없다.

## 권장 실험 순서

```bash
# 1. base 1.5B held-out baseline
python3 evaluate_exact_match.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir full_v3 \
  --output reports/v3_base_1.5b.jsonl

# 2. 50-step smoke
python3 train_qlora.py --config configs/qlora_v3_smoke_1.5b.json

# 3. full 1.5B
python3 train_qlora.py --config configs/qlora_v3_full_1.5b.json
python3 evaluate_exact_match.py \
  --adapter-path outputs/v3-full-1.5b \
  --data-dir full_v3 \
  --output reports/v3_qlora_1.5b.jsonl

# 4. full 7B
python3 train_qlora.py --config configs/qlora_v3_full_7b.json
python3 evaluate_exact_match.py \
  --adapter-path outputs/v3-full-7b \
  --data-dir full_v3 \
  --output reports/v3_qlora_7b.jsonl
```

학습은 conversational prompt-completion 형식, assistant completion-only loss, 4-bit NF4, all-linear LoRA를 사용한다. 평가는 전체 exact match와 scenario/profile별 exact match를 출력한다.

## 현재 상태와 한계

- 스키마·누출·split·episode replay 검증: 완료
- 1.5B tokenizer 사전 점검: 완료
- 실제 weight update: NVIDIA GPU에서 실행 필요
- 실제 vCAN/ISO-TP trace: 미포함
- ECU profile: 합성 profile 3개
- timing, P2/P2*, session timeout: 미포함

따라서 v3는 미관측 합성 ECU 동작에 대한 초기 일반화 실험용이며 실제 ECU 일반화를 입증하지 않는다.
