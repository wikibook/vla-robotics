# 01-train-smolvla: SmolVLA 5,000 step 파인튜닝 예제

## 프로젝트 설명

`ch07/01-train-smolvla`의 학습량을 25배(200 → 5,000 step) 늘린 변형입니다. 학습 step이 늘어남에 따라 정책의 블록 들어올리기 능력이 어떻게 변하는지 [`02-eval-smolvla`](../02-eval-smolvla/)로 정량 비교하기 위해 분리했습니다. (Python 3.12 / LeRobot 0.6.0)

```text
LeRobotDataset 로드 → 사전학습 SmolVLA 로드 → 옵티마이저 → 학습 루프 → 체크포인트
```

- [train.py](train.py) — 전체 학습 파이프라인 (ch07/01의 `train.py`와 상수 5개만 다름)

| 상수 | ch07/01 | ch08/01 | 의미 |
|------|--------:|--------:|------|
| `TOTAL_STEPS` | 200 | 5,000 | 데이터셋 약 2.1 epoch (50 ep × 9,428 frame, batch=4) |
| `WARMUP_STEPS` | 20 | 500 | `TOTAL_STEPS`의 10% 유지 |
| `LOG_EVERY` | 10 | 100 | 긴 학습 로그 축소 |
| `CKPT_EVERY` | 100 | 1,000 | 중간 ckpt 4개 + 최종 |
| `OUT_DIR` 끝부분 | `soarm-200-run1` | `soarm-5000-run1` | 학습 step별 디렉터리 분리 |

나머지(`LR=1e-4`, `BATCH_SIZE=4`, `GRAD_CLIP=1.0`, `WEIGHT_DECAY=1e-4`, 데이터셋, 베이스 체크포인트 `lerobot/smolvla_base`)는 ch07/01과 동일합니다.

## 실행 방법

```bash
uv sync
uv run python train.py
```

진행 상황은 100 step 단위로 출력됩니다. VRAM 점유는 batch=4 그대로면 ch07/01과 동일합니다(12 GB에서 안정 동작).

## 무엇이 출력되나

`ch07/01`과 같은 형식으로 100 step 간격 로그가 나오고, 1,000 step마다 중간 체크포인트가 저장됩니다. 끝나면 `./checkpoints/soarm-5000-run1/`에 `step_1000`~`step_4000`과 `step_final` 다섯 개가 남습니다. [`02-eval-smolvla`](../02-eval-smolvla/)의 `eval.py`가 기본으로 `step_final`을 참조하므로 학습 종료 후 별도 설정 없이 평가할 수 있습니다.

체크포인트 다섯 개가 모두 생겼으면 정상입니다. 학습 시간은 장비에 따라 크게 다르며, RTX 3060 12GB에서 수십 분 규모입니다. 학습량을 늘렸을 때 성능이 어떻게 변하는지는 `02-eval-smolvla`로 측정하고, 그 해석은 책 8장에서 다룹니다.
