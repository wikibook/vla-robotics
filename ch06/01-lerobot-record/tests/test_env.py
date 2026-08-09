"""ch06 SoArm101Env MuJoCo 통합 검증."""

import numpy as np
import pytest

from env import (
    BLOCK_X_RANGE,
    BLOCK_Y_RANGE,
    GRASP_ASSIST_CLOSE_QPOS,
    IMG_SIZE,
    N_DOF,
    SIDE_IMG_SIZE,
    STATE_HIGH,
    STATE_LOW,
    SoArm101Env,
)


@pytest.fixture
def so_arm_env():
    # 1. 환경 생성·종료
    env = SoArm101Env(control_hz=20)
    yield env
    env.close()


@pytest.mark.mujoco
def test_reset_returns_expected_obs_shapes(so_arm_env: SoArm101Env):
    # 1. reset 관측 shape·dtype
    obs = so_arm_env.reset()
    assert obs["image"].shape == (IMG_SIZE, IMG_SIZE, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["side_image"].shape == (SIDE_IMG_SIZE, SIDE_IMG_SIZE, 3)
    assert obs["side_image"].dtype == np.uint8
    assert obs["state"].shape == (6,)
    assert obs["state"].dtype == np.float32


@pytest.mark.mujoco
def test_step_observation_unchanged_when_zero_action(so_arm_env: SoArm101Env):
    # 1. 0 액션 1스텝 → state 거의 동일
    obs0 = so_arm_env.reset()
    obs1, _ = so_arm_env.step(np.zeros(N_DOF, dtype=np.float32))
    np.testing.assert_allclose(obs0["state"], obs1["state"], atol=1e-3)


@pytest.mark.mujoco
def test_step_clamps_action_within_state_bounds(so_arm_env: SoArm101Env):
    # 1. 큰 액션 100스텝 후 ctrl 클램프
    so_arm_env.reset()
    big = np.full(N_DOF, 10.0, dtype=np.float32)
    for _ in range(100):
        so_arm_env.step(big)
    ctrl = so_arm_env.data.ctrl[:N_DOF]
    assert np.all(ctrl >= STATE_LOW)
    assert np.all(ctrl <= STATE_HIGH)


@pytest.mark.mujoco
def test_set_grasp_pins_block_to_gripper(so_arm_env: SoArm101Env):
    # 1. grasp 활성 + 그리퍼 닫힘 → 블록 z == grasp site z
    so_arm_env.reset()
    so_arm_env.set_grasp(True)
    grip_idx = int(so_arm_env.model.joint("gripper").qposadr[0])
    so_arm_env.data.qpos[grip_idx] = GRASP_ASSIST_CLOSE_QPOS - 0.05
    so_arm_env.data.ctrl[5] = so_arm_env.data.qpos[grip_idx]
    so_arm_env.step(np.zeros(N_DOF, dtype=np.float32))

    block_id = so_arm_env.model.body("block").id
    block_z = float(so_arm_env.data.xpos[block_id][2])
    grasp_z = float(so_arm_env.data.site_xpos[so_arm_env._grasp_site_id][2])
    assert block_z == pytest.approx(grasp_z, abs=1e-4)


@pytest.mark.mujoco
def test_randomize_block_pose_within_range(so_arm_env: SoArm101Env):
    # 1. 100회 reset → 블록 (x,y) 범위
    adr = so_arm_env._block_qposadr
    for _ in range(100):
        so_arm_env.reset()
        x = float(so_arm_env.data.qpos[adr])
        y = float(so_arm_env.data.qpos[adr + 1])
        assert BLOCK_X_RANGE[0] <= x <= BLOCK_X_RANGE[1]
        assert BLOCK_Y_RANGE[0] <= y <= BLOCK_Y_RANGE[1]


@pytest.mark.mujoco
def test_block_pinched_initially_false(so_arm_env: SoArm101Env):
    # 1. reset 직후 block_pinched False
    so_arm_env.reset()
    assert so_arm_env.block_pinched() is False
