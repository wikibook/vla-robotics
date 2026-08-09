"""ch04 FK·PD 순수 함수 검증 (main import — main() 미호출)."""

import math

import pytest

from main import BASE_HEIGHT, KD, KP, L1, L2, forward_kinematics, pd_torque


def test_fk_zero_pose():
    # 1. 영자세 FK
    x, z = forward_kinematics(0.0, 0.0)
    assert x == pytest.approx(0.0)
    assert z == pytest.approx(L1 + L2 + BASE_HEIGHT)


def test_fk_known_angles():
    # 1. q1=π/2, q2=0 알려진 좌표
    x, z = forward_kinematics(math.pi / 2, 0.0)
    assert x == pytest.approx(-(L1 + L2))
    assert z == pytest.approx(BASE_HEIGHT)


def test_fk_smooth_continuity():
    # 1. 미세 각도 변화 → 미세 좌표 변화
    q1, q2 = 0.1, 0.2
    x0, z0 = forward_kinematics(q1, q2)
    eps = 1e-6
    x1, z1 = forward_kinematics(q1 + eps, q2)
    x2, z2 = forward_kinematics(q1, q2 + eps)
    assert abs(x1 - x0) < 1e-3
    assert abs(z1 - z0) < 1e-3
    assert abs(x2 - x0) < 1e-3
    assert abs(z2 - z0) < 1e-3


def test_pd_torque_sign():
    # 1. 양의 위치 오차 → 양의 토크
    # 2. 양의 속도 → 감쇠 음수 항
    tau_pos = pd_torque(1.0, 0.0, 0.0)
    assert tau_pos > 0.0
    assert tau_pos == pytest.approx(KP * 1.0)
    tau_vel = pd_torque(0.0, 0.0, 0.5)
    assert tau_vel < 0.0
    assert tau_vel == pytest.approx(-KD * 0.5)
