"""MuJoCo 뷰어 실행 예제."""

from pathlib import Path

import mujoco
from mujoco import viewer

# 1. 모델·데이터 로드
xml_path = Path(__file__).parent / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 2. 뷰어 실행
viewer.launch(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
)
