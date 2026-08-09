"""ch04 IK 순수 함수 검증 (FK 헬퍼는 동일 상수로 자급)."""

import math

import pytest

from main import BASE_HEIGHT, L1, L2, Q1_MAX, Q1_MIN, Q2_MAX, Q2_MIN, inverse_kinematics


def _forward_kinematics(q1: float, q2: float) -> tuple[float, float]:
    # 1. ch04/03과 동일 FK (테스트 자급)
    x = -(L1 * math.sin(q1) + L2 * math.sin(q1 + q2))
    z = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + BASE_HEIGHT
    return x, z


def test_ik_returns_none_for_unreachable():
    # 1. 반지름 > L1+L2 → None
    assert inverse_kinematics(0.0, 2.0, 0.0, 0.0) is None


def test_ik_returns_none_outside_joint_limits():
    # 1. 리밋 밖 자세에 해당하는 좌표 → None
    x, z = _forward_kinematics(math.pi / 2, 0.0)
    assert inverse_kinematics(x, z, 0.0, 0.0) is None


@pytest.mark.parametrize(
    ("q1", "q2"),
    [
        (0.0, 0.0),
        (0.2, 0.3),
        (-0.3, 0.4),
        (math.pi / 6, -math.pi / 6),
    ],
)
def test_ik_roundtrip_with_fk(q1: float, q2: float):
    # 1. FK → IK → 원래 관절 근사
    x, z = _forward_kinematics(q1, q2)
    sol = inverse_kinematics(x, z, q1, q2)
    assert sol is not None
    q1_out, q2_out = sol
    assert q1_out == pytest.approx(q1, abs=1e-4)
    assert q2_out == pytest.approx(q2, abs=1e-4)


def test_ik_seed_disambiguates_elbow():
    # 1. 동일 (x,z)에 seed에 따라 elbow 해 선택
    q1_seed, q2_up = 0.2, 0.55
    q2_down = -0.55
    x, z = _forward_kinematics(q1_seed, q2_up)
    sol_up = inverse_kinematics(x, z, q1_seed, q2_up)
    sol_down = inverse_kinematics(x, z, q1_seed, q2_down)
    assert sol_up is not None
    assert sol_down is not None
    assert sol_up[1] == pytest.approx(q2_up, abs=0.05)
    assert sol_down[1] == pytest.approx(q2_down, abs=0.05)
    assert sol_up[1] * sol_down[1] < 0.0


def test_ik_boundary_cos_q2_clipping():
    # 1. cos_q2 = +1 신장 경계 — NaN 없음
    z_ext = L1 + L2 + BASE_HEIGHT
    sol_ext = inverse_kinematics(0.0, z_ext, 0.0, 0.0)
    assert sol_ext is not None
    assert not math.isnan(sol_ext[0])
    assert not math.isnan(sol_ext[1])

    # 2. 신장 직전(클리핑 구간) — NaN 없음
    sol_near = inverse_kinematics(0.0, z_ext - 0.001, 0.0, 0.0)
    assert sol_near is not None
    assert not math.isnan(sol_near[0])

    # 3. elbow-down 깊은 굽힘 — 리밋 내 해 존재
    q1, q2 = 0.1, -0.7
    x, z = _forward_kinematics(q1, q2)
    sol_bent = inverse_kinematics(x, z, q1, q2)
    assert sol_bent is not None
    assert Q1_MIN <= sol_bent[0] <= Q1_MAX
    assert Q2_MIN <= sol_bent[1] <= Q2_MAX


def test_ik_rejects_solution_outside_xml_joint_range():
    # 1. q2가 scene.xml range="-45 45" 밖인 자세는 두 해 모두 리밋 밖 → None
    x, z = _forward_kinematics(0.1, -1.4)
    assert inverse_kinematics(x, z, 0.1, -1.4) is None
