"""SO-ARM 101 scripted pick 정책 예제."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

from env import JOINT_NAMES, SoArm101Env

# 1. 그리퍼·블록 식별자
GRIPPER_SITE = "gripperframe"
GRIPPER_BODY = "gripper"
BLOCK_BODY = "block"


# 2. 픽업 FSM 단계
class PickStage(IntEnum):
    PRE_OPEN = -1
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    RETREAT = 4


# 3. 픽업 FSM 하이퍼파라미터
@dataclass(frozen=True)
class PickFsmConfig:
    approach_height: float = 0.10
    grasp_height: float = 0.012
    lift_height: float = 0.22
    grip_open: float = 1.62
    grip_closed: float = -0.1
    wrist_roll_target: float = math.pi / 2
    pre_open_steps: int = 8
    pos_tol: float = 0.02
    approach_xy_tol: float = 0.015
    descend_z_tol: float = 0.010
    approach_timeout: int = 100
    descend_timeout: int = 80
    stall_dq_eps: float = 5e-4
    stall_steps: int = 4
    grasp_contact_streak: int = 3
    grasp_stage_timeout: int = 60
    grasp_hold_steps: int = 10
    grip_close_qpos: float = 0.9
    lift_timeout: int = 80
    lift_success_z: float = 0.12
    max_dq: float = 0.04
    descend_max_dq: float = 0.02
    lift_max_dq: float = 0.015
    ik_damping: float = 0.1


# 4. 한 에피소드 최대 스텝 예산
PICK_FSM = PickFsmConfig()
EPISODE_STEP_BUDGET = (
    PICK_FSM.pre_open_steps
    + PICK_FSM.approach_timeout
    + PICK_FSM.descend_timeout
    + PICK_FSM.grasp_stage_timeout
    + PICK_FSM.grasp_hold_steps
    + PICK_FSM.lift_timeout
    + 40
)


class ScriptedPickPolicy:
    """접근 -> 하강 -> 파지 -> 들어올리기 순서로 블록 집기."""

    def __init__(self, env: SoArm101Env, cfg: PickFsmConfig | None = None) -> None:
        self.env = env
        self._cfg = cfg or PICK_FSM
        self.model = env.model
        self.data = env.data
        self.action_scale = env.action_scale

        self.site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            GRIPPER_SITE,
        )
        self.gripper_body_id = self.model.body(GRIPPER_BODY).id
        self.block_id = self.model.body(BLOCK_BODY).id
        self._arm_dof = np.array(
            [self.model.joint(name).dofadr[0] for name in JOINT_NAMES[:4]],
            dtype=np.int64,
        )
        self._wrist_roll_ctrl_idx = 4
        self._grip_ctrl_idx = 5
        self.reset_episode()

    def reset_episode(self) -> None:
        self.save_requested = False
        self._stage = PickStage.PRE_OPEN
        self._timer = 0
        self._lift_xy: np.ndarray | None = None
        self._post_grasp_hold = 0
        self._contact_streak = 0
        self._grasp_confirmed = False
        self._last_dq_norm = 0.0
        self._stall_counter = 0
        self.last_grasp_cause = ""

    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def _block_target(self, block_xyz: np.ndarray, z: float) -> np.ndarray:
        return np.array([block_xyz[0], block_xyz[1], z])

    def _xy_error(self, current_xyz: np.ndarray, target_xyz: np.ndarray) -> float:
        return float(np.linalg.norm(current_xyz[:2] - target_xyz[:2]))

    def _reached_xy(self, current_xyz: np.ndarray, target_xyz: np.ndarray) -> bool:
        return self._xy_error(current_xyz, target_xyz) < self._cfg.approach_xy_tol

    def _timed_out(self, timeout: int) -> bool:
        return self._timer >= timeout

    def _solve_ik_delta(
        self,
        position_error: np.ndarray,
        jacobian: np.ndarray,
        max_dq: float,
    ) -> np.ndarray:
        damping = self._cfg.ik_damping
        system = jacobian @ jacobian.T + (damping**2) * np.eye(3)
        try:
            rhs = np.linalg.solve(system, position_error)
        except np.linalg.LinAlgError:
            rhs = np.linalg.lstsq(system, position_error, rcond=None)[0]

        dq_arm = jacobian.T @ rhs
        dq_norm = float(np.linalg.norm(dq_arm))
        if dq_norm > max_dq:
            dq_arm *= max_dq / dq_norm
            dq_norm = max_dq
        self._last_dq_norm = dq_norm
        return dq_arm

    def _update_stall(self) -> bool:
        if self._timer > 3 and self._last_dq_norm < self._cfg.stall_dq_eps:
            self._stall_counter += 1
        else:
            self._stall_counter = 0
        return self._stall_counter >= self._cfg.stall_steps

    def _make_action(
        self,
        target_xyz: np.ndarray,
        grip_target: float,
        max_dq: float | None = None,
    ) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        position_error = target_xyz - self._ee_pos()

        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model,
            self.data,
            jacp,
            None,
            self._ee_pos(),
            self.gripper_body_id,
        )
        dq_arm = self._solve_ik_delta(
            position_error,
            jacp[:, self._arm_dof],
            self._cfg.max_dq if max_dq is None else float(max_dq),
        )

        action = np.zeros(6, dtype=np.float32)
        action[:4] = (dq_arm / self.action_scale).astype(np.float32)
        action[self._wrist_roll_ctrl_idx] = self._wrist_roll_action()
        action[self._grip_ctrl_idx] = self._grip_action(grip_target)
        return np.clip(action, -1.0, 1.0)

    def _wrist_roll_action(self) -> float:
        wrist_roll = float(self.data.ctrl[self._wrist_roll_ctrl_idx])
        return (self._cfg.wrist_roll_target - wrist_roll) / self.action_scale

    def _grip_action(self, grip_target: float) -> float:
        grip = float(self.data.ctrl[self._grip_ctrl_idx])
        return (grip_target - grip) / self.action_scale

    def _gripper_qpos(self) -> float:
        return float(self.data.qpos[int(self.model.joint("gripper").qposadr[0])])

    def _advance_stage(self, next_stage: PickStage) -> None:
        self._stage = next_stage
        self._timer = 0
        self._stall_counter = 0

    def _hold_wrist_and_gripper(self, grip_target: float) -> np.ndarray:
        action = np.zeros(6, dtype=np.float32)
        action[self._wrist_roll_ctrl_idx] = self._wrist_roll_action()
        action[self._grip_ctrl_idx] = self._grip_action(grip_target)
        return np.clip(action, -1.0, 1.0)

    def _pre_open_action(self) -> np.ndarray:
        self._timer += 1
        if self._gripper_qpos() >= self._cfg.grip_open - 0.2 or self._timed_out(
            self._cfg.pre_open_steps
        ):
            self._advance_stage(PickStage.APPROACH)

        return self._hold_wrist_and_gripper(self._cfg.grip_open)

    def _approach_action(self, block_xyz: np.ndarray, ee_xyz: np.ndarray) -> np.ndarray:
        target = self._block_target(block_xyz, self._cfg.approach_height)
        self._timer += 1
        if (
            self._reached_xy(ee_xyz, target)
            or self._update_stall()
            or self._timed_out(self._cfg.approach_timeout)
        ):
            self._advance_stage(PickStage.DESCEND)
        return self._make_action(target, self._cfg.grip_open)

    def _descend_action(self, block_xyz: np.ndarray, ee_xyz: np.ndarray) -> np.ndarray:
        target = self._block_target(block_xyz, self._cfg.grasp_height)
        self._timer += 1
        z_error = float(abs(ee_xyz[2] - target[2]))
        reached = self._reached_xy(ee_xyz, target) and z_error < self._cfg.descend_z_tol
        if (
            reached
            or self._update_stall()
            or self._timed_out(self._cfg.descend_timeout)
        ):
            self._advance_stage(PickStage.GRASP)
        return self._make_action(
            target,
            self._cfg.grip_open,
            max_dq=self._cfg.descend_max_dq,
        )

    def _update_grasp_confirmation(self) -> None:
        if (
            self._gripper_qpos() <= self._cfg.grip_close_qpos
            and self.env.block_pinched()
        ):
            self._contact_streak += 1
        else:
            self._contact_streak = 0

        if self._contact_streak >= self._cfg.grasp_contact_streak:
            self._grasp_confirmed = True
        if self._grasp_confirmed:
            self._post_grasp_hold += 1

    def _grasp_ready_to_lift(self) -> bool:
        return self._post_grasp_hold >= self._cfg.grasp_hold_steps

    def _start_lift(self, ee_xyz: np.ndarray) -> None:
        self._stage = PickStage.LIFT
        self._timer = 0
        self._post_grasp_hold = 0
        self._lift_xy = ee_xyz[:2].copy()
        self.env.set_grasp(True)
        self.last_grasp_cause = "contact"

    def _fail_grasp(self) -> None:
        self._stage = PickStage.RETREAT
        self.last_grasp_cause = "failed_no_contact"

    def _grasp_action(self, block_xyz: np.ndarray, ee_xyz: np.ndarray) -> np.ndarray:
        target = self._block_target(block_xyz, self._cfg.grasp_height)
        self._timer += 1
        self._update_grasp_confirmation()

        if self._grasp_ready_to_lift():
            self._start_lift(ee_xyz)
        elif (
            self._timed_out(self._cfg.grasp_stage_timeout) and not self._grasp_confirmed
        ):
            self._fail_grasp()

        return self._make_action(target, self._cfg.grip_closed)

    def _retreat_action(self, block_xyz: np.ndarray) -> np.ndarray:
        self.env.set_grasp(False)
        target = self._block_target(block_xyz, self._cfg.approach_height)
        return self._make_action(target, self._cfg.grip_open)

    def _lift_action(self, ee_xyz: np.ndarray) -> np.ndarray:
        assert self._lift_xy is not None
        target = np.array([self._lift_xy[0], self._lift_xy[1], self._cfg.lift_height])
        self._timer += 1
        reached = np.linalg.norm(ee_xyz - target) < self._cfg.pos_tol
        if reached or self._timer >= self._cfg.lift_timeout:
            self._finish_lift()
        return self._make_action(
            target,
            self._cfg.grip_closed,
            max_dq=self._cfg.lift_max_dq,
        )

    def _finish_lift(self) -> None:
        block_z = float(self.data.xpos[self.block_id][2])
        if block_z >= self._cfg.lift_success_z:
            self.save_requested = True
            self.last_grasp_cause = "lifted"
            return
        self.env.set_grasp(False)
        self._stage = PickStage.RETREAT
        self.last_grasp_cause = "lift_slipped"

    def get_action(self) -> np.ndarray:
        block_xyz = self.data.xpos[self.block_id].copy()
        ee_xyz = self._ee_pos()

        # 1. PRE_OPEN: 그리퍼 사전 개방
        if self._stage == PickStage.PRE_OPEN:
            return self._pre_open_action()
        # 2. APPROACH: 블록 위 접근 자세 정렬
        if self._stage == PickStage.APPROACH:
            return self._approach_action(block_xyz, ee_xyz)
        # 3. DESCEND: 블록 위로 하강
        if self._stage == PickStage.DESCEND:
            return self._descend_action(block_xyz, ee_xyz)
        # 4. GRASP: 그리퍼 닫고 접촉 streak 확인
        if self._stage == PickStage.GRASP:
            return self._grasp_action(block_xyz, ee_xyz)
        # 5. RETREAT: 파지 실패 시 후퇴
        if self._stage == PickStage.RETREAT:
            return self._retreat_action(block_xyz)
        # 6. LIFT: 블록 들어 올리기
        return self._lift_action(ee_xyz)
