"""ch06 ScriptedPickPolicy MuJoCo 통합 검증."""

import numpy as np
import pytest

from env import SoArm101Env
from scripted import (
    EPISODE_STEP_BUDGET,
    PICK_FSM,
    PickFsmConfig,
    PickStage,
    ScriptedPickPolicy,
)


@pytest.fixture
def so_arm_env():
    env = SoArm101Env(control_hz=20)
    yield env
    env.close()


@pytest.fixture
def pick_policy(so_arm_env: SoArm101Env):
    policy = ScriptedPickPolicy(so_arm_env)
    so_arm_env.reset()
    policy.reset_episode()
    return policy


@pytest.mark.mujoco
def test_get_action_returns_clipped_6dof(pick_policy: ScriptedPickPolicy):
    # 1. 첫 호출 action shape·범위
    action = np.asarray(pick_policy.get_action(), dtype=np.float32)
    assert action.shape == (6,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


@pytest.mark.mujoco
def test_episode_step_budget_consistency():
    # 1. EPISODE_STEP_BUDGET ≥ 단계 timeout 합
    cfg = PICK_FSM
    min_budget = (
        cfg.pre_open_steps
        + cfg.approach_timeout
        + cfg.descend_timeout
        + cfg.grasp_stage_timeout
        + cfg.grasp_hold_steps
        + cfg.lift_timeout
    )
    assert EPISODE_STEP_BUDGET >= min_budget


@pytest.mark.mujoco
def test_failed_grasp_sets_retreat_cause(so_arm_env: SoArm101Env):
    # 1. grasp_stage_timeout 강제 → failed_no_contact
    cfg = PickFsmConfig(grasp_stage_timeout=0)
    policy = ScriptedPickPolicy(so_arm_env, cfg=cfg)
    so_arm_env.reset()
    policy.reset_episode()
    policy._stage = PickStage.GRASP

    block_xyz = policy.data.xpos[policy.block_id].copy()
    ee_xyz = policy._ee_pos()
    policy._grasp_action(block_xyz, ee_xyz)

    assert policy._stage == PickStage.RETREAT
    assert policy.last_grasp_cause == "failed_no_contact"


@pytest.mark.mujoco
@pytest.mark.slow
def test_fsm_advances_through_stages(so_arm_env: SoArm101Env):
    # 1. FSM 단계 순회
    policy = ScriptedPickPolicy(so_arm_env)
    so_arm_env.reset()
    policy.reset_episode()
    stages_seen: set[PickStage] = set()

    for _ in range(EPISODE_STEP_BUDGET):
        stages_seen.add(policy._stage)
        action = policy.get_action()
        so_arm_env.step(action)
        if policy.save_requested:
            break

    for stage in (
        PickStage.PRE_OPEN,
        PickStage.APPROACH,
        PickStage.DESCEND,
        PickStage.GRASP,
    ):
        assert stage in stages_seen


@pytest.mark.mujoco
@pytest.mark.slow
def test_save_requested_set_on_lift_success(so_arm_env: SoArm101Env):
    # 1. 들어올리기 성공 시 save_requested
    policy = ScriptedPickPolicy(so_arm_env)
    so_arm_env.reset()
    policy.reset_episode()
    policy._stage = PickStage.LIFT
    policy._lift_xy = policy._ee_pos()[:2].copy()
    so_arm_env.set_grasp(True)

    block_id = policy.block_id
    policy.data.xpos[block_id][2] = PICK_FSM.lift_success_z + 0.05
    policy._finish_lift()

    assert policy.save_requested is True
    assert policy.last_grasp_cause == "lifted"
