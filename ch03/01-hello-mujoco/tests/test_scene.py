"""ch03 hello 장면·MuJoCo 로드 검증 (main.py 미 import — 뷰어 자동 실행 방지)."""

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


def test_scene_xml_gravity_plane_sphere():
    # 1. option 중력 z 음수
    # 2. plane·sphere geom 존재
    root = ET.parse(_SCENE_XML).getroot()
    assert root.tag == "mujoco"
    option = root.find("option")
    assert option is not None
    gravity = option.get("gravity")
    assert gravity is not None
    gz = float(gravity.split()[2])
    assert gz < 0
    planes = root.findall(".//geom[@type='plane']")
    spheres = root.findall(".//geom[@type='sphere']")
    assert len(planes) >= 1
    assert len(spheres) >= 1


@pytest.mark.mujoco
def test_mjmodel_loads_from_xml():
    # 1. XML에서 MjModel 로드
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    assert model.ngeom >= 2


@pytest.mark.mujoco
def test_gravity_z_matches_scene():
    # 1. 컴파일된 중력 z
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    assert model.opt.gravity[2] == pytest.approx(-9.81)


@pytest.mark.mujoco
def test_sphere_initial_height_then_falls():
    # 1. 구 body 초기 높이
    # 2. 스텝 후 z 감소
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sphere_geom = -1
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            sphere_geom = gid
            break
    assert sphere_geom >= 0
    body_id = int(model.geom_bodyid[sphere_geom])
    z0 = float(data.xpos[body_id, 2])
    assert z0 == pytest.approx(5.0, abs=1e-2)
    for _ in range(80):
        mujoco.mj_step(model, data)
    z1 = float(data.xpos[body_id, 2])
    assert z1 < z0


@pytest.mark.mujoco
def test_step_advances_time():
    # 1. mj_step 후 시뮬 시간 증가
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    t0 = float(data.time)
    mujoco.mj_step(model, data)
    assert float(data.time) > t0
