# VLA 로보틱스 예제 코드

MuJoCo 시뮬레이션과 LeRobot으로 VLA(Vision-Language-Action) 로보틱스를 실습하는 예제 모음입니다.

## 예제 구성

| 디렉터리 | 주제 |
|---|---|
| [ch03/01-hello-mujoco](ch03/01-hello-mujoco/) | MuJoCo 첫 실행 — 중력과 자유 낙하 |
| [ch04/01-robotarm-mjdata](ch04/01-robotarm-mjdata/) | mjData로 로봇 상태 읽기 |
| [ch04/02-robotarm-control](ch04/02-robotarm-control/) | 관절 모터와 PD 제어 |
| [ch04/03-robotarm-kinematics](ch04/03-robotarm-kinematics/) | 순운동학(FK)으로 손끝 좌표 계산 |
| [ch04/04-robotarm-ik](ch04/04-robotarm-ik/) | 역운동학(IK)으로 좌표 기반 이동 |
| [ch05/01-lerobot-pusht](ch05/01-lerobot-pusht/) | LeRobot Diffusion Policy PushT 평가 |
| [ch06/01-lerobot-record](ch06/01-lerobot-record/) | SO-ARM 101 시연 데이터 수집 |
| [ch07/01-train-smolvla](ch07/01-train-smolvla/) | SmolVLA 파인튜닝 |
| [ch07/02-infer-smolvla](ch07/02-infer-smolvla/) | SmolVLA 추론 |
| [ch08/01-train-smolvla](ch08/01-train-smolvla/) | 학습량 5,000 step 확장 |
| [ch08/02-eval-smolvla](ch08/02-eval-smolvla/) | 성공률 정량 평가 |
| [ch09/01-data-augment](ch09/01-data-augment/) | offline 데이터 증강 |
| [ch09/02-train-eval](ch09/02-train-eval/) | 증강 데이터 학습·평가·비교 |
| [lelab](lelab/) | SO-ARM 101 실물 실습 도구 (벤더 복사본) |

## 실행 방법

모든 예제는 독립된 uv 패키지입니다. 예제 디렉터리로 이동해 실행합니다.

```bash
cd ch03/01-hello-mujoco
uv sync
uv run python main.py
```

예제별 실행 방법과 출력 확인 요령은 각 디렉터리의 `README.md`에 있습니다.

각 README는 "어떻게 실행하고 무엇을 확인하는가"만 다룹니다. 성공률·학습 시간 같은 측정값과 그 해석은 책 본문에 있습니다. 학습과 정책 평가에는 난수가 들어가 실행할 때마다 숫자가 달라지므로, README의 수치를 재현 목표로 삼지 않도록 의도적으로 분리했습니다.

## 요구 환경

- 파트 1~3: Ubuntu 22.04 (WSL2 포함)
- 파트 4 실물 로봇: 네이티브 Windows 또는 Linux (WSL2 비권장)
- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/) — 의존성 설치와 실행에 사용
- 학습·평가 예제는 CUDA GPU 권장 (실측 환경: RTX 3060 12GB)

## 라이선스

이 저장소의 예제 코드는 별도 표시가 없는 한 [Apache License 2.0](LICENSE)으로 배포됩니다.

`lelab/`은 Hugging Face LeLab의 벤더 복사본이며 자체 `LICENSE`와 `VENDOR_SOURCE.txt`를 포함합니다. **이 복사본은 상류 원본에서 수정됐습니다** — 무엇을 고쳤는지는 `lelab/VENDOR_SOURCE.txt`의 Modifications 절에 있습니다. 이 복사본의 문제는 상류가 아니라 이 저장소로 알려 주세요.

SO-ARM101 URDF/STL 자산과 주요 외부 의존성 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 확인하세요.
