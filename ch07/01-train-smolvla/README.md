# 01-train-smolvla: SmolVLA 파인튜닝 예제

## 프로젝트 설명

6장에서 모은 SO-ARM 101 시연 데이터로 SmolVLA를 200 step 부분 파인튜닝하는 예제입니다. (Python 3.12 / LeRobot 0.6.0)

```text
LeRobotDataset 로드 → 사전학습 SmolVLA 로드 → 옵티마이저 → 학습 루프 → 체크포인트
```

- [train.py](train.py) — 전체 학습 파이프라인

데이터셋은 6장 업로드 결과인 공개 저장소 `makepluscode/so_arm101_block_picking_main`(50 에피소드 / 9,428 프레임)이라 별도 인증 없이 받아집니다.

## 실행 방법

```bash
uv sync
uv run python train.py
```

`DATASET_REPO_ID`, `BASE_CHECKPOINT`, `LR`, `BATCH_SIZE`, `TOTAL_STEPS` 등 핵심 값은 모두 `train.py` 상단 상수로 노출됩니다. GPU 메모리(VRAM)가 부족하면 `BATCH_SIZE`를 줄이거나 입력 이미지 해상도를 낮춥니다.

## 무엇이 출력되나

허브 데이터셋을 내려받은 뒤 학습 대상 파라미터 규모(전체 450.0M 중 99.9M — vision freeze + expert only)를 알리고, 10 step 간격으로 `[step NNNN] loss=... lr=...`을 출력합니다. 100 step마다 중간 체크포인트를, 끝나면 최종 체크포인트를 `./checkpoints/soarm-200-run1/`에 저장하고 총 학습 시간을 표시합니다. 추론 예제 `02-infer-smolvla`가 이 경로를 그대로 입력으로 씁니다.

정상 동작은 이렇게 확인합니다. warmup 구간(step 0~20)에서 lr이 1e-4까지 올랐다가 cosine decay로 내려가고, loss가 1점대에서 시작해 0.x대로 내려간 뒤 그 부근에서 오르내리면 됩니다. batch=4라 loss가 매 줄 단조 감소하지 않는 것이 정상입니다. 마지막에 `step_final` 디렉터리가 생겼는지 확인하면 파이프라인이 끝까지 돈 것입니다.

구체적인 loss 값과 학습 시간은 실행할 때마다 달라집니다. 가중치 초기화와 데이터 순서에 난수가 들어가기 때문이며, 학습 시간은 장비에 따라서도 크게 달라집니다. 200 step은 수렴 구간이 아니라 파이프라인 점검 구간이라는 점도 함께 기억해 두시면 됩니다. 손실 곡선 해석은 책 7장에서 다룹니다.
