# 01-lerobot-record: SO-ARM 101 시연 데이터 수집 예제

## 프로젝트 설명

MuJoCo 속 SO-ARM 101이 블록을 집는 scripted 시연을 만들고, 그 결과를 LeRobotDataset 형식으로 저장하는 예제입니다. (Python 3.12 / LeRobot 0.6.0)

```text
환경 reset → scripted policy로 action 생성 → MuJoCo step → frame 누적 → episode 저장
```

- [main.py](main.py) — 전체 실행 흐름
- [env.py](env.py) — MuJoCo 환경과 관측 생성
- [scripted.py](scripted.py) — IK 기반 집기 정책
- [scene.xml](scene.xml) — SO-ARM 101 씬

## 실행 방법

```bash
uv sync
uv run python main.py
uv run python main.py 5
```

인자를 주지 않으면 `SOARM101_NUM_EPISODES` 또는 기본값 1을, 인자를 주면 저장할 에피소드 수를 직접 지정합니다. MuJoCo는 헤드리스(EGL)로 동작하므로 창은 열리지 않고 카메라 이미지는 오프스크린 렌더링으로만 수집됩니다.

## 무엇이 출력되나

성공적으로 들어올리기까지 끝난 에피소드만 `./datasets/local/so_arm101_block_picking_main`에 저장됩니다. 데이터셋에는 이미지 키 `observation.images.top`·`observation.images.side`, 상태 키 `observation.state`, 행동 키 `action`이 담깁니다.

reset 때마다 블록의 초기 위치를 x `0.345~0.395`, y `-0.045~0.045` 범위에서 작게 무작위화해 데이터 다양성을 확보합니다. 로봇 베이스 yaw는 고정입니다.
