# Colab QLoRA Runbook

GPU 런타임에서 아래 순서대로 실행하면 된다. 첫 실행은 smoke test까지만 진행한다.

## 1. GPU 확인

```bash
!nvidia-smi
```

CUDA GPU가 표시되지 않으면 Colab 메뉴에서 GPU 런타임으로 변경한다.

## 2. 저장소와 패키지 준비

```bash
!git clone https://github.com/LSC18/uds-nrc-finetuning-dataset.git
%cd uds-nrc-finetuning-dataset
!python -m pip install -r requirements-train.txt
```

## 3. 데이터와 런타임 사전 점검

```bash
!python scripts/validate_dataset.py
!shasum -a 256 -c SHA256SUMS
!python scripts/preflight.py
```

통과 기준:

- `status`: `ready`
- `truncation_count`: `0`
- train / validation / test: `3904 / 431 / 386`
- SHA-256 파일 5개: 모두 `OK`

## 4. 학습 전 baseline 기록

전체 386개 평가 전에 20개만 실행해 generation 경로를 확인한다.

```bash
!python evaluate_exact_match.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --limit 20 \
  --output reports/baseline_20.jsonl
```

문제가 없으면 `--limit` 없이 전체 baseline을 기록한다.

## 5. 50-step smoke test

```bash
!python train_qlora.py --config configs/qlora_smoke.json
```

완료 조건:

- `outputs/uds-nrc-smoke/adapter_config.json` 생성
- `outputs/uds-nrc-smoke/final_metrics.json` 생성
- train/eval loss가 `NaN`이 아님

## 6. smoke adapter 평가

```bash
!python evaluate_exact_match.py \
  --adapter-path outputs/uds-nrc-smoke \
  --output reports/smoke_test_predictions.jsonl
```

baseline과 smoke adapter의 `exact_match`를 비교한다.

## 7. 전체 학습

smoke test가 정상일 때만 실행한다.

```bash
!python train_qlora.py --config configs/qlora_full.json
!python evaluate_exact_match.py \
  --adapter-path outputs/uds-nrc-full \
  --output reports/full_test_predictions.jsonl
```

## 8. 보존할 결과

Colab 런타임 종료 전에 다음 항목을 Drive 또는 별도 저장소에 보존한다.

```text
outputs/uds-nrc-smoke/
outputs/uds-nrc-full/
reports/baseline_20.jsonl
reports/smoke_test_predictions.jsonl
reports/full_test_predictions.jsonl
```

최종 보고에는 base model, GPU, package versions, seed, train/eval loss, exact-match, 실행 시간, 실패 scenario를 기록한다.
