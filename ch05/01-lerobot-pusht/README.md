# 01-lerobot-pusht: Diffusion Policy PushT 평가 예제

## 프로젝트 설명

LeRobot의 사전 학습된 Diffusion Policy 모델을 PushT 시뮬레이션 환경에서 10 에피소드 평가하는 예제입니다. LeRobot 추론 파이프라인의 전체 흐름을 보여 줍니다. (Python 3.12 / LeRobot 0.6.0)

```text
환경 관측(obs) → 키 변환 → 전처리 → 정책 추론 → 후처리 → 환경 제어(step)
```

사전 학습 모델은 [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht), 환경은 `gym_pusht/PushT-v0`(에피소드당 300스텝), 데이터셋은 [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht)입니다.

## 실행 방법

```bash
uv sync
uv run python main.py
```

## 무엇이 출력되나

먼저 데이터셋 정보(에피소드 206개, 프레임 25,650개)가 나오고, 이어서 10 에피소드를 하나씩 평가하며 에피소드별 최대보상과 성공 여부를 출력합니다. 성공 기준은 최대보상 0.90 이상이며, 마지막에 성공 횟수와 평균보상이 요약됩니다. 진행 과정 GIF와 최대 보상 시점 PNG는 `snapshots/`에 저장됩니다.

성공 횟수는 실행할 때마다 달라집니다. Diffusion Policy는 잡음에서 출발해 행동을 생성하는데 이 예제는 파이토치 난수를 고정하지 않기 때문입니다. 에피소드 시드는 0~9로 고정돼 블록 배치는 매번 같지만, 같은 배치에서도 정책이 다른 궤적을 그립니다. 보상이 0.00 부근부터 1.00 가까이까지 흩어지는 것이 정상이며, 이것이 정책 평가에 다수 에피소드 반복이 필요한 이유입니다. 성능 수치와 그 해석은 책 5장에서 다룹니다.

실행이 정상인지는 데이터셋이 206 에피소드로 로드되는지, 10 에피소드가 끝까지 돌며 `snapshots/`에 파일이 쌓이는지로 판단하면 됩니다.
