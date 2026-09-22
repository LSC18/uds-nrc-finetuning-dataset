# Colab QLoRA Runbook — v3

## 1. GPU 확인 및 설치

```bash
!nvidia-smi
!git clone https://github.com/LSC18/uds-nrc-finetuning-dataset.git
%cd uds-nrc-finetuning-dataset
!python -m pip uninstall -y -q torchvision torchaudio torchtext
!python -m pip install -r requirements-colab.txt
```

## 2. 데이터·런타임 검증

```bash
!python scripts/validate_dataset.py --data-dir full_v3
!shasum -a 256 -c SHA256SUMS
!python scripts/preflight.py \
  --config configs/qlora_v3_smoke_1.5b.json \
  --report reports/preflight_v3_1.5b_gpu.json
```

통과 기준:

- `status`: `ready`
- `truncation_count`: `0`
- train / validation / test: `8266 / 777 / 1364`
- GPU 이름과 CUDA 사용 가능 여부 기록

## 3. 1.5B base baseline

```bash
!python evaluate_exact_match.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir full_v3 \
  --output reports/v3_base_1.5b.jsonl
```

## 4. 50-step smoke test

```bash
!python train_qlora.py --config configs/qlora_v3_smoke_1.5b.json
!python evaluate_exact_match.py \
  --adapter-path outputs/v3-smoke-1.5b \
  --data-dir full_v3 \
  --output reports/v3_smoke_1.5b.jsonl
```

완료 조건은 adapter와 `final_metrics.json` 생성, train/eval loss 비-NaN이다.

## 5. 1.5B 전체 학습

```bash
!python train_qlora.py --config configs/qlora_v3_full_1.5b.json
!python evaluate_exact_match.py \
  --adapter-path outputs/v3-full-1.5b \
  --data-dir full_v3 \
  --output reports/v3_qlora_1.5b.jsonl
```

## 6. 7B 비교 실험

7B는 24GB GPU를 권장한다. 16GB T4에서는 batch 1, gradient checkpointing 설정으로 시도하되 OOM이면 evaluation batch와 max length를 먼저 낮춘다.

```bash
!python scripts/preflight.py \
  --config configs/qlora_v3_full_7b.json \
  --report reports/preflight_v3_7b_gpu.json
!python evaluate_exact_match.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --data-dir full_v3 \
  --output reports/v3_base_7b.jsonl
!python train_qlora.py --config configs/qlora_v3_full_7b.json
!python evaluate_exact_match.py \
  --adapter-path outputs/v3-full-7b \
  --data-dir full_v3 \
  --output reports/v3_qlora_7b.jsonl
```

## 7. 보존할 결과

```text
outputs/v3-smoke-1.5b/
outputs/v3-full-1.5b/
outputs/v3-full-7b/
reports/preflight_v3_*_gpu.json
reports/v3_base_*.jsonl
reports/v3_qlora_*.jsonl
```

보고서에는 모델, GPU, 패키지 버전, seed, 총 학습시간, train/eval loss, 전체 exact-match, scenario별 exact-match, held-out profile exact-match를 기록한다.
