"""ch04 PD 제어·장면 검증 (main.py 미 import)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from main import KD, KP, pd_torque

# 1. scene.xml 경로
_SCENE_DIR = Path(__file__).resolve().parent.parent
_SCENE_XML = _SCENE_DIR / "scene.xml"


def test_scene_xml_exists():
    # 1. scene.xml 파일 존재
    assert _SCENE_XML.is_file()


def test_scene_xml_has_actuators():
    # 1. 모터(actuator) 2개
    root = ET.parse(_SCENE_XML).getroot()
    motors = root.findall(".//actuator/motor")
    assert len(motors) == 2


def test_pd_torque_sign():
    # 1. 양의 위치 오차 → 양의 토크
    # 2. 양의 속도 → 감쇠 음수 항
    assert pd_torque(1.0, 0.0, 0.0) == pytest.approx(KP)
    assert pd_torque(0.0, 0.0, 0.5) == pytest.approx(-KD * 0.5)


@pytest.mark.mujoco
def test_mjmodel_has_two_actuators():
    # 1. nu == 2
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    assert model.nu == 2
