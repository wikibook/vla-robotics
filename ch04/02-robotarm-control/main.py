"""로봇팔 PD 제어 예제."""

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


def print_state(data: mujoco.MjData, ee_id: int) -> None:
    """현재 관절 각도·속도·토크·end_effector 위치를 한 줄로 출력"""
    q1 = math.degrees(data.qpos[0])
    q2 = math.degrees(data.qpos[1])
    v1 = math.degrees(data.qvel[0])
    v2 = math.degrees(data.qvel[1])
    ee = data.xpos[ee_id]
    print(
        f"[t={data.time:4.1f}s] "
        f"q=({q1:+5.1f},{q2:+5.1f})° "
        f"v=({v1:+6.1f},{v2:+6.1f})°/s  "
        f"ctrl=({data.ctrl[0]:+6.2f},{data.ctrl[1]:+6.2f})Nm  "
        f"ee=({ee[0]:+.2f},{ee[2]:+.2f})"
    )


def main() -> None:
    # 2. 모델·데이터 생성
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    # 3. keyframe 초기화
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 4. end_effector body ID 조회
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

    with viewer.launch_passive(model, data) as v:
        last_print_time = -1.0
        current_phase = 0

        # 5. 단계별 제어 루프
        while v.is_running() and data.time < 11.0:
            loop_start = time.time()

            # 5-1. 단계 1 목표 토크 계산
            if data.time < 4.0:
                phase = 1
                data.ctrl[0] = pd_torque(0.0, data.qpos[0], data.qvel[0])
                data.ctrl[1] = pd_torque(0.0, data.qpos[1], data.qvel[1])

            # 5-2. 단계 2 목표 토크 계산
            elif data.time < 8.0:
                phase = 2
                tgt1 = 0.35 * math.sin(1.5 * data.time)  # ≈ ±20°
                tgt2 = 0.17 * math.sin(1.5 * data.time)  # ≈ ±10°
                data.ctrl[0] = pd_torque(tgt1, data.qpos[0], data.qvel[0])
                data.ctrl[1] = pd_torque(tgt2, data.qpos[1], data.qvel[1])

            # 5-3. 단계 3 토크 제거
            else:
                phase = 3
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0

            # 5-4. 단계 전환 출력
            if phase != current_phase:
                labels = {
                    1: "PD 제어 → ㄱ자에서 수직으로 들어올리기",
                    2: "사인파 추종 → 0° 근처 왕복 운동",
                    3: "토크 제거 → 반대쪽으로 낙하",
                }
                print(f"\n{'=' * 60}")
                print(f"  단계 {phase}: {labels[phase]}")
                print(f"{'=' * 60}")
                current_phase = phase

            mujoco.mj_step(model, data)

            # 5-5. 0.5초 마다 상태 출력
            if data.time - last_print_time >= 0.5:
                print_state(data, ee_id)
                last_print_time = data.time

            # 5-6. 뷰어 상태 동기화
            v.sync()

            # 5-7. 실시간 속도 보정
            elapsed = time.time() - loop_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
