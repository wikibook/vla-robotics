"""ch04 mjData 장면·MuJoCo 로드 검증 (main.py 미 import)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# 1. scene.xml 경로
_SCENE_DIR = Path(__file__).resolve().parent.parent
_SCENE_XML = _SCENE_DIR / "scene.xml"


def test_scene_xml_exists():
    # 1. scene.xml 파일 존재
    assert _SCENE_XML.is_file()


def test_scene_xml_parses():
    # 1. Well-formed XML
    ET.parse(_SCENE_XML)


def test_scene_xml_robot_structure():
    # 1. 2축 로봇·end_effector·keyframe
    root = ET.parse(_SCENE_XML).getroot()
    assert root.tag == "mujoco"
    option = root.find("option")
    assert option is not None
    gz = float(option.get("gravity", "0 0 0").split()[2])
    assert gz < 0
    joints = root.findall(".//joint[@type='hinge']")
    assert len(joints) == 2
    assert root.find(".//body[@name='end_effector']") is not None
    assert root.find(".//key[@name='init']") is not None


@pytest.mark.mujoco
def test_mjmodel_loads_from_xml():
    # 1. XML에서 MjModel 로드
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    assert model.nq >= 2
    assert model.nu == 0


@pytest.mark.mujoco
def test_keyframe_reset_sets_qpos():
    # 1. keyframe 초기 qpos
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    assert float(data.qpos[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(data.qpos[1]) == pytest.approx(0.0873, abs=1e-4)


@pytest.mark.mujoco
def test_step_advances_time():
    # 1. mj_step 후 시뮬 시간 증가
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    t0 = float(data.time)
    mujoco.mj_step(model, data)
    assert float(data.time) > t0
