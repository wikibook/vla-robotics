# 02-eval-smolvla: SmolVLA 정량 평가 예제

## 프로젝트 설명

학습된 SmolVLA 체크포인트를 MuJoCo `SoArm101Env` 위에서 N 에피소드 굴려 블록 들어올리기 성공률을 측정하는 예제입니다. (Python 3.12 / LeRobot 0.6.0)

```text
시드 고정 → N 에피소드 추론 → 에피소드별 블록 z 기록 → 임계값 판정 → 통계·그래프
```

- [eval.py](eval.py) — 전체 평가 파이프라인

`SoArm101Env`와 `scene.xml`은 `ch06/01-lerobot-record/`의 자산을 `sys.path` 등록으로 재사용합니다. 성공 판정은 블록 z가 임계 `LIFT_SUCCESS_Z = 0.12 m`를 넘고 마지막 10프레임 중 5프레임 이상 유지될 때입니다. 블록 초기 z가 약 0.007 m이므로 약 11 cm 이상 부양해야 성공입니다.

## 실행 방법

```bash
uv sync
uv run python eval.py
```

기본값으로 `../01-train-smolvla/checkpoints/soarm-5000-run1/step_final`을 평가합니다. 다른 체크포인트를 보려면 `eval.py` 상단 `CKPT_PATH`만 바꿉니다. `N_EPISODES`, `MAX_STEPS`, `SEED`도 모두 상단 상수로 노출됩니다. 결과는 `eval_results.json`과 `images/success_rate.png`로 저장되며 매 실행마다 덮어써집니다.

## 무엇이 출력되나

20 에피소드를 하나씩 돌며 `[ep NN] success=... max_z=... final_z=... grasp=...`을 출력하고, 마지막에 성공률·grasp 활성 수·평균 max z를 요약합니다. 요약은 `eval_results.json`에, 그래프는 `images/success_rate.png`에 저장되며 매 실행마다 덮어써집니다.

`SEED=42`가 고정하는 것은 환경 난수기, 즉 20개 블록 초기 위치뿐입니다. 파이토치 난수는 고정하지 않으므로 같은 체크포인트라도 회차마다 성공 개수가 달라집니다. 20 에피소드에서 1 에피소드가 5%p라 편차가 작지 않습니다. 한 번의 숫자보다 여러 번 돌린 분포로 읽어야 하며, 그 해석은 책 8장에서 다룹니다.

읽는 요령은 이렇습니다. max z가 블록 정지 높이(≈0.007 m)에 몰려 있으면 정책이 접근조차 못 하는 상태, 0.05~0.10 m로 흩어지면 블록은 건드리나 들어올리지 못하는 상태, grasp 활성 수가 성공 수보다 크면 잡는 동작은 학습됐으나 lift 단계가 불완전한 상태입니다. 성공 임계는 max z 0.12 m입니다.

`CKPT_PATH`를 중간 체크포인트로 바꿔 가며 실행하면 학습량에 따른 곡선을 직접 그려 볼 수 있습니다. 책 8장은 체크포인트 5개를 3회씩 평가한 결과를 [표 8.6]에 싣고 있으며, 9장 `compare.py`는 그 값을 기준선으로 삼습니다.
