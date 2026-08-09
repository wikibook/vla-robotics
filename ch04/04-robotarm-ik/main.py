"""로봇팔 IK 제어 예제."""

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

# 3. 관절 범위 (scene.xml 의 joint range="-45 45" 와 동일)
Q1_MIN, Q1_MAX = -math.pi / 4, math.pi / 4
Q2_MIN, Q2_MAX = -math.pi / 4, math.pi / 4

# 4. IK 경계 허용 오차
_IK_EPS = 1e-9


def inverse_kinematics(
    x_target: float,
    z_target: float,
    q1_seed: float,
    q2_seed: float,
) -> tuple[float, float] | None:
    """2링크 팔(plane IK): 목표 (x,z) → 관절 (q1,q2), rad

    - 2차원 평면 + 회전축 정의에 맞춘 해석적(analytic) 공식
    - elbow(팔꿈치) 위/아래 해 2개 중, limit 통과 + seed 근접 1개 선택
    - 도달 불가·리밋 밖이면 None
    """
    # 1. 목표 좌표 평면 변환
    px = -x_target
    pz = z_target - BASE_HEIGHT

    # 2. 어깨-손목 거리 계산
    r_sq = px * px + pz * pz

    # 3. 코사인 법칙 기반 q2 계산
    cos_q2 = (r_sq - L1 * L1 - L2 * L2) / (2 * L1 * L2)

    # 4. 도달 가능 범위 검사
    if cos_q2 < -1.0 - _IK_EPS or cos_q2 > 1.0 + _IK_EPS:
        return None

    # 5. acos 입력 범위 보정
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2_plus = math.acos(cos_q2)

    # 6. 두 elbow 해 후보 생성
    candidates: list[tuple[float, float]] = []
    for q2_try in (q2_plus, -q2_plus):
        # 6-1. q1 계산
        k1 = L1 + L2 * math.cos(q2_try)
        k2 = L2 * math.sin(q2_try)
        q1_try = math.atan2(px, pz) - math.atan2(k2, k1)

        # 6-2. 리밋 내부 해만 유지
        if Q1_MIN <= q1_try <= Q1_MAX and Q2_MIN <= q2_try <= Q2_MAX:
            candidates.append((q1_try, q2_try))

    if not candidates:
        return None

    # 7. seed 근접 해 선택
    return min(
        candidates,
        key=lambda q: (q[0] - q1_seed) ** 2 + (q[1] - q2_seed) ** 2,
    )


def get_step1_target() -> tuple[float, float]:
    """단계 1 IK 목표 좌표 (x, z)

    - 두 관절이 모두 굽어야 닿는 지점
    - 수직 자세 (0, 0.8) 은 IK 해가 (0°, 0°) 라 PD 예제와 결과 동일
    - elbow 위/아래 두 해 모두 리밋 안 → seed 선택 확인 가능
    """
    return -0.05, 0.76


def print_state(
    data: mujoco.MjData,
    ee_id: int,
    x_tgt: float,
    z_tgt: float,
) -> None:
    """현재 관절 각도 + 목표 좌표 + MuJoCo xpos 출력"""
    q1, q2 = data.qpos[0], data.qpos[1]
    ee = data.xpos[ee_id]
    pos_err = math.hypot(x_tgt - ee[0], z_tgt - ee[2])
    print(
        f"[t={data.time:4.1f}s] "
        f"목표=({x_tgt:+.3f},{z_tgt:.3f})  "
        f"q=({math.degrees(q1):+6.1f},{math.degrees(q2):+6.1f})°  "
        f"xpos=({ee[0]:+.3f},{ee[2]:+.3f})  "
        f"pos_err={pos_err:.4f}"
    )


def main() -> None:
    # 5. 모델·데이터 생성
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    # 6. keyframe 초기화
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 7. end_effector body ID 조회
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

    x_tgt, z_tgt = get_step1_target()
    sol = inverse_kinematics(x_tgt, z_tgt, data.qpos[0], data.qpos[1])
    if not sol:
        print("초기 IK: 도달 불가 (발생하지 않아야 함)")
        return
    q1_tgt, q2_tgt = sol

    print(f"\nIK 목표=({x_tgt:.3f}, {z_tgt:.3f})")
    print(f"  q=({math.degrees(q1_tgt):.1f}°, {math.degrees(q2_tgt):.1f}°)")
    print(f"\n{'=' * 60}")
    print("  단계 1: IK 목표 (-0.05, 0.76) 굽힌 자세로 수렴")
    print(f"{'=' * 60}")

    last_print_time = -1.0
    t_end = 10.0
    with viewer.launch_passive(model, data) as v:
        # 8. IK 목표 추종 루프
        while v.is_running() and data.time < t_end:
            loop_start = time.time()

            # 8-1. PD 토크 적용
            data.ctrl[0] = pd_torque(q1_tgt, data.qpos[0], data.qvel[0])
            data.ctrl[1] = pd_torque(q2_tgt, data.qpos[1], data.qvel[1])

            mujoco.mj_step(model, data)

            # 8-2. 0.5초 마다 상태 출력
            if data.time - last_print_time >= 0.5:
                print_state(data, ee_id, x_tgt, z_tgt)
                last_print_time = data.time

            # 8-3. 뷰어 상태 동기화
            v.sync()

            # 8-4. 실시간 속도 보정
            elapsed = time.time() - loop_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
