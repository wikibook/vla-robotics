"""로봇팔 FK 검증 예제."""

import math
import time

import mujoco
from mujoco import viewer

# 1. PD 제어 게인
KP = 40.0  # 위치 오차 반응 강도
KD = 15.0  # 속도 감쇠 강도


def pd_torque(target: float, q: float, v: float) -> float:
    """PD 제어: τ = Kp·(목표−현재) + Kd·(0−속도)"""
    return KP * (target - q) + KD * (0.0 - v)


# 2. 링크 길이
L1 = 0.4  # base → link2 관절
L2 = 0.3  # link2 관절 → end_effector
BASE_HEIGHT = 0.1  # base 높이 + link1 시작 높이


def forward_kinematics(q1: float, q2: float) -> tuple[float, float]:
    """정기구학(FK): 관절 각도(rad) → 손끝 좌표 (x, z)

    axis="0 -1 0" 이므로 양의 각도가 -x 방향으로 회전.
    base 높이(0.1)를 z에 더해 월드 좌표와 맞춤.
    """
    x = -(L1 * math.sin(q1) + L2 * math.sin(q1 + q2))
    z = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + BASE_HEIGHT
    return x, z


def print_state(data: mujoco.MjData, ee_id: int) -> None:
    """현재 관절 각도 + FK 좌표 vs MuJoCo xpos 비교 출력"""
    q1, q2 = data.qpos[0], data.qpos[1]
    fk_x, fk_z = forward_kinematics(q1, q2)
    ee = data.xpos[ee_id]
    err = math.hypot(fk_x - ee[0], fk_z - ee[2])
    print(
        f"  [t={data.time:5.1f}s] "
        f"q=({math.degrees(q1):+6.1f},{math.degrees(q2):+6.1f})°  "
        f"FK=({fk_x:+.3f},{fk_z:+.3f})  "
        f"xpos=({ee[0]:+.3f},{ee[2]:+.3f})  "
        f"err={err:.4f}"
    )


def main() -> None:
    # 3. 모델·데이터 생성
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    # 4. keyframe 초기화
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 5. end_effector body ID 조회
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

    # 6. 초기 FK 검증
    q1, q2 = data.qpos[0], data.qpos[1]
    fk_x, fk_z = forward_kinematics(q1, q2)
    ee = data.xpos[ee_id]
    err = math.hypot(fk_x - ee[0], fk_z - ee[2])
    print(
        f"초기 FK 검증: q=({math.degrees(q1):.1f}°, {math.degrees(q2):.1f}°) → FK=({fk_x:.3f}, {fk_z:.3f})"
    )
    print(f"  xpos: ({ee[0]:.3f}, {ee[2]:.3f})  err={err:.4f}")

    with viewer.launch_passive(model, data) as v:
        last_print_time = -1.0
        current_phase = 0

        # 7. 단계별 제어 루프
        while v.is_running() and data.time < 11.0:
            loop_start = time.time()

            # 7-1. 단계 1 목표 토크 계산
            if data.time < 4.0:
                phase = 1
                data.ctrl[0] = pd_torque(0.0, data.qpos[0], data.qvel[0])
                data.ctrl[1] = pd_torque(0.0, data.qpos[1], data.qvel[1])

            # 7-2. 단계 2 목표 토크 계산
            elif data.time < 8.0:
                phase = 2
                tgt1 = 0.35 * math.sin(1.5 * data.time)  # ≈ ±20°
                tgt2 = 0.17 * math.sin(1.5 * data.time)  # ≈ ±10°
                data.ctrl[0] = pd_torque(tgt1, data.qpos[0], data.qvel[0])
                data.ctrl[1] = pd_torque(tgt2, data.qpos[1], data.qvel[1])

            # 7-3. 단계 3 토크 제거
            else:
                phase = 3
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0

            # 7-4. 단계 전환 출력
            if phase != current_phase:
                labels = {
                    1: "PD 제어 → ㄱ자→수직 수렴 + FK 검증",
                    2: "사인파 추종 → 왕복 운동 + FK 추적",
                    3: "토크 제거 → 중력 낙하 + FK 추적",
                }
                print(f"\n{'=' * 60}")
                print(f"  단계 {phase}: {labels[phase]}")
                print(f"{'=' * 60}")
                current_phase = phase

            mujoco.mj_step(model, data)

            # 7-5. 0.5초 마다 상태 출력
            if data.time - last_print_time >= 0.5:
                print_state(data, ee_id)
                last_print_time = data.time

            # 7-6. 뷰어 상태 동기화
            v.sync()

            # 7-7. 실시간 속도 보정
            elapsed = time.time() - loop_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
