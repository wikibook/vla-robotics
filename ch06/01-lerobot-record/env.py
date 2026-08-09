"""SO-ARM 101 MuJoCo 환경 예제."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

IMG_SIZE = 128
SIDE_IMG_SIZE = 256
N_DOF = 6
CAMERA_NAME = "top"
SIDE_CAMERA_NAME = "side"
SCENE_PATH = Path(__file__).parent / "scene.xml"
ACTION_SCALE = 0.05

STATE_LOW = np.array(
    [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453],
    dtype=np.float32,
)
STATE_HIGH = np.array(
    [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
    dtype=np.float32,
)
BLOCK_X_RANGE = (0.345, 0.395)
BLOCK_Y_RANGE = (-0.045, 0.045)
GRASP_ASSIST_XY_TOL = 0.025
GRASP_ASSIST_Z_TOL = 0.035
GRASP_ASSIST_CLOSE_QPOS = 0.9
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class SoArm101Env:
    """scripted policy가 사용하는 최소 환경 API."""

    def __init__(self, control_hz: int = 20) -> None:
        self.action_scale = ACTION_SCALE
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)

        self._sim_steps = max(
            1,
            int(round((1.0 / control_hz) / self.model.opt.timestep)),
        )
        self._home_key = self.model.key("home").id
        self._block_qposadr = self.model.joint("block_free").qposadr[0]
        self._block_dofadr = self.model.joint("block_free").dofadr[0]
        self._qpos_idx = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINT_NAMES],
            dtype=np.int64,
        )
        self._grasp_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "gripperframe",
        )
        self._static_finger_geom = self._find_mesh_geom_id(
            "gripper",
            "wrist_roll_follower_so101_v1",
        )
        self._moving_jaw_geom = self._find_mesh_geom_id(
            "moving_jaw_so101_v1",
            "moving_jaw_so101_v1",
        )

        self._renderer = mujoco.Renderer(
            self.model,
            height=IMG_SIZE,
            width=IMG_SIZE,
        )
        self._camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            CAMERA_NAME,
        )
        self._side_renderer = mujoco.Renderer(
            self.model,
            height=SIDE_IMG_SIZE,
            width=SIDE_IMG_SIZE,
        )
        self._side_camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            SIDE_CAMERA_NAME,
        )

        self._grasp_enabled = False
        self._rng = np.random.default_rng()

    def _current_state(self) -> np.ndarray:
        return self.data.qpos[self._qpos_idx].astype(np.float32).copy()

    def _render_image(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._camera_id)
        return self._renderer.render()

    def _render_side_image(self) -> np.ndarray:
        self._side_renderer.update_scene(self.data, camera=self._side_camera_id)
        return self._side_renderer.render()

    def _make_observation(self) -> dict[str, Any]:
        return {
            "image": self._render_image(),
            "side_image": self._render_side_image(),
            "state": self._current_state(),
        }

    def _randomize_block_pose(self) -> None:
        adr = self._block_qposadr
        self.data.qpos[adr : adr + 3] = [
            float(self._rng.uniform(*BLOCK_X_RANGE)),
            float(self._rng.uniform(*BLOCK_Y_RANGE)),
            0.008,
        ]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _reset_controls_to_home(self) -> None:
        self.data.ctrl[:N_DOF] = self.data.qpos[self._qpos_idx]

    def _apply_action(self, action: np.ndarray) -> None:
        target = self.data.ctrl[:N_DOF] + self.action_scale * action
        self.data.ctrl[:N_DOF] = np.clip(target, STATE_LOW, STATE_HIGH)

    def _pin_block_to_gripper(self) -> None:
        adr = self._block_qposadr
        self.data.qpos[adr : adr + 3] = self.data.site_xpos[self._grasp_site_id]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._block_dofadr : self._block_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_grasp_weld(self) -> None:
        if not self._grasp_enabled:
            return
        if float(self.data.qpos[self._qpos_idx[5]]) <= GRASP_ASSIST_CLOSE_QPOS:
            self._pin_block_to_gripper()

    def reset(self) -> dict[str, Any]:
        # 1. 홈 자세 초기화
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)

        # 2. 블록 위치 무작위 배치
        self._randomize_block_pose()

        # 3. grasp 상태 해제
        self.set_grasp(False)

        # 4. ctrl 초기 자세 정렬
        self._reset_controls_to_home()
        mujoco.mj_forward(self.model, self.data)
        return self._make_observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], bool]:
        action = np.asarray(action, dtype=np.float32).reshape(N_DOF)

        # 1. action 누적 후 ctrl 갱신
        self._apply_action(action)

        # 2. 제어 주기 동안 물리 sub-step 반복
        for _ in range(self._sim_steps):
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_weld()
        return self._make_observation(), False

    def set_grasp(self, active: bool) -> None:
        """두 턱이 동시에 블록을 물었을 때만 보조 고정 허용."""
        self._grasp_enabled = bool(active)

    def _find_mesh_geom_id(self, body_name: str, mesh_name: str) -> int:
        body_id = self.model.body(body_name).id
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] != body_id:
                continue
            if self.model.geom_group[geom_id] != 3:
                continue
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            try:
                if self.model.mesh(mesh_id).name == mesh_name:
                    return geom_id
            except Exception:
                continue
        raise RuntimeError(f"mesh {mesh_name} not found on body {body_name}")

    def block_pinched(self) -> bool:
        """블록이 양쪽 턱에 동시에 닿았는지 검사."""
        block_body_id = self.model.body("block").id
        touching_static = False
        touching_moving = False

        # 1. contact 목록 순회
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            if body1 == block_body_id:
                other_geom = geom2
            elif body2 == block_body_id:
                other_geom = geom1
            else:
                continue

            if other_geom == self._static_finger_geom:
                touching_static = True
            elif other_geom == self._moving_jaw_geom:
                touching_moving = True

        # 2. 양쪽 턱 동시 접촉 판정
        if touching_static and touching_moving:
            return True

        # 3. 근접 거리 폴백 판정
        block_xyz = self.data.xpos[block_body_id]
        gripper_xyz = self.data.site_xpos[self._grasp_site_id]
        xy_error = float(np.linalg.norm(block_xyz[:2] - gripper_xyz[:2]))
        z_error = float(abs(block_xyz[2] - gripper_xyz[2]))
        return xy_error <= GRASP_ASSIST_XY_TOL and z_error <= GRASP_ASSIST_Z_TOL

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass
        try:
            self._side_renderer.close()
        except Exception:
            pass
