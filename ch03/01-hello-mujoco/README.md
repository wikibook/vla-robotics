# 01-hello-mujoco: MuJoCo Hello 예제

## 프로젝트 설명

MuJoCo를 처음 써 볼 때 보는 가장 단순한 예제입니다. 바닥(plane geom) 하나와 그 위에서 중력(gravity)에 의해 떨어지는 공(sphere) 하나만 있습니다. 공 body에는 `freejoint`를 넣어 3D 공간에서 자유롭게 움직이도록 했습니다.

## 실행 방법

```bash
uv sync
uv run python main.py
```

## 실행 결과

MuJoCo 뷰어가 열리고, 공이 바닥으로 떨어져 닿은 뒤 멈추는 모습이 보입니다. 배경은 light gray이며 좌우 UI는 숨겨져 3D 뷰만 표시됩니다.

![실행 결과](screenshot.png)
