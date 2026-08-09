# 02-train-eval: 증강 데이터 학습 · 평가 · ch08 비교

## 프로젝트 설명

[`01-data-augment`](../01-data-augment/)에서 만든 200 ep 데이터셋으로 SmolVLA 파인튜닝(fine-tuning)을 20,000 step 수행하고, 4k/8k/12k/16k/20k 5단계 체크포인트를 8장과 동일 지표(성공률 · avg max z · grasp)로 비교하는 예제입니다. (Python 3.12 / LeRobot 0.6.0)

```text
01-data-augment (200ep)
    → train.py        20k step, ckpt 4k 간격
    → eval.py         5단계 정량 평가
    → compare.py      ch08 5단계 vs ch09
```

- [train.py](train.py) — SmolVLA 20k 파인튜닝
- [eval.py](eval.py) — 4k/8k/12k/16k/20k 5단계 평가
- [compare.py](compare.py) — ch08 3-run 평균과 표·그래프

데이터가 4배가 되었으므로 학습 step도 4배로 스케일합니다(batch=4, 200 ep 기준 1 epoch ≈ 4,716 step). 평가 기준은 [`ch08/02-eval-smolvla`](../../ch08/02-eval-smolvla/)와 같은 블록 z ≥ 0.12 m, 20 ep, `SEED=42`, `MAX_STEPS=200`입니다.

## 실행 방법

선행 조건으로 [`01-data-augment`](../01-data-augment/)의 `augment_dataset.py`가 완료되어 있어야 합니다.

```bash
uv sync
uv run python train.py
uv run python eval.py
uv run python compare.py
```

`train.py`는 약 155분, `eval.py`는 약 11분, `compare.py`는 1초 정도 걸립니다.

## 무엇이 출력되나

`train.py`는 100 step 간격 로그를 내고 4,000 step마다 체크포인트를 저장해, `./checkpoints/soarm-aug200-20k/`에 `step_4000`부터 `step_20000`까지 다섯 개를 남깁니다.

`eval.py`는 이 5단계를 순서대로 평가하며 단계별 성공률·avg max z·grasp 활성 수를 출력하고, 요약을 `eval.json`에 저장합니다. 평가 조건은 8장과 같은 20 에피소드 × 200 step, `SEED=42`입니다. 같은 조건을 유지해야 성공률 차이를 평가 방식이 아니라 정책 차이로 읽을 수 있습니다.

`compare.py`는 `eval.json`을 읽어 두 실험의 5단계를 나란히 놓은 표(`compare.md`)와 그래프(`images/compare.png`)를 만듭니다. 8장 쪽은 책 [표 8.6]의 측정값을 `compare.py` 상단 `CH08_BASELINE` 상수로 두었습니다. 저자가 측정한 기준선이며, 직접 재평가하면 다른 수가 나옵니다.

성공률은 회차마다 달라집니다. 시드는 블록 배치만 고정하고 파이토치 난수는 고정하지 않기 때문입니다. 증강 데이터가 성능에 어떤 영향을 주는지, 최종 체크포인트가 최고 성능인지 같은 해석은 책 9장에서 다룹니다.
