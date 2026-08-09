"""MuJoCo 상태 출력 예제."""

import time

import mujoco
from mujoco import viewer


def print_state(data: mujoco.MjData, ee_id: int) -> None:
    # 1. end_effector 위치와 관절 상태 문자열 정리
    ee_pos = data.xpos[ee_id]
    qpos_str = "[" + ", ".join(f"{x:.3f}" for x in data.qpos) + "]"
    qvel_str = "[" + ", ".join(f"{x:.3f}" for x in data.qvel) + "]"
    print(
        f"[t={data.time:5.1f}] "
        f"qpos={qpos_str} "
        f"qvel={qvel_str} "
        f"ee=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
    )


def main() -> None:
    # 1. 모델·데이터 생성
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    # 2. keyframe 초기화
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 3. end_effector body ID 조회
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

    with viewer.launch_passive(model, data) as v:
        # 4. 상태 출력 주기 변수
        last_print_time = -1.0

        # 5. 5초 동안 시뮬레이션 반복
        while v.is_running() and data.time < 5.0:
            loop_start = time.time()
            mujoco.mj_step(model, data)

            # 5-1. 1초 마다 상태 출력
            if data.time - last_print_time >= 1.0:
                print_state(data, ee_id)
                last_print_time = data.time

            # 5-2. 뷰어 상태 동기화
            v.sync()

            # 5-3. 실시간 속도 보정
            elapsed = time.time() - loop_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
