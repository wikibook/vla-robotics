# 02-infer-smolvla: SmolVLA 첫 추론 예제

## 프로젝트 설명

`01-train-smolvla`에서 만든 체크포인트로 MuJoCo `SoArm101Env`에서 한 에피소드를 실행하는 예제입니다. (Python 3.12 / LeRobot 0.6.0)

```text
환경 reset → 관측 전처리 → policy.select_action → 역정규화 → env.step
```

- [infer.py](infer.py) — 전체 추론 루프

`SoArm101Env`와 `scene.xml`은 `ch06/01-lerobot-record/`의 자산을 `sys.path` 등록으로 그대로 재사용합니다.

## 실행 방법

```bash
uv sync
uv run python infer.py
```

체크포인트 경로(`CKPT_PATH`), 지시문(`TASK_INSTRUCTION`), 추론 길이(`MAX_STEPS`)는 모두 `infer.py` 상단 상수로 노출됩니다.

## 무엇이 출력되나

체크포인트 경로를 알린 뒤 10 step 간격으로 `[t=NNN] action=[6축] qpos[0:3]=[...]`을 출력하고, 마지막에 총 소요 시간을 표시합니다.

액션 값은 앞 예제에서 학습한 체크포인트에 따라 달라지므로 매번 다르게 나옵니다. 확인할 것은 값 자체가 아니라 형태입니다. 초반 몇 스텝의 움직임이 부자연스러우면 역정규화·통계 파일 연결을, 블록에 접근하지만 잡지 못하면 데이터 부족이나 그리퍼 차원 학습 부족을, 매번 같은 방향으로만 움직이면 특정 모드에 빠진 상태를 의심합니다.

소요 시간은 `CONTROL_HZ=20` 기준 이론 최소치가 10초이고, 여기에 모델 추론 시간과 초기화 시간이 더해집니다. 실시간 20Hz 제어를 만족하는지는 이 차이로 판단합니다. 정량 평가는 8장 예제에서 다룹니다.
