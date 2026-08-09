#!/usr/bin/env python3

"""SmolVLA 정량 평가 예제."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

# 1. ch06 SoArm101Env 재사용 경로 등록
CH06_DIR = Path(__file__).resolve().parents[2] / "ch06" / "01-lerobot-record"
sys.path.insert(0, str(CH06_DIR))

from env import SoArm101Env  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

# 2. 평가 설정 (CKPT_PATH 만 바꾸면 다른 학습 결과도 평가 가능)
CKPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "01-train-smolvla/checkpoints/soarm-5000-run1/step_final"
)
TASK_INSTRUCTION = "pick the block"
N_EPISODES = 20
MAX_STEPS = 200
CONTROL_HZ = 20
SEED = 42

# 3. 성공 판정 기준
LIFT_SUCCESS_Z = 0.12  # 블록이 이 높이 이상이면 들어올림 인정 (scripted policy 기준)
LIFT_HOLD_FRAMES = 5  # 마지막 10 프레임 중 임계 충족 최소 프레임 수
GRASP_CLOSE_QPOS = 0.9  # 그리퍼가 이 값 이하이면 닫힘 판정
GRASP_CONTACT_STREAK = 3  # 양쪽 턱 동시 접촉 연속 프레임 임계값

# 4. 결과 저장 경로
OUT_DIR = Path(__file__).resolve().parent
RESULT_JSON = OUT_DIR / "eval_results.json"
PLOT_PATH = OUT_DIR / "images" / "success_rate.png"

CAMERA_KEY = "observation.images.top"
SIDE_CAMERA_KEY = "observation.images.side"
STATE_KEY = "observation.state"


def to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    # 1. 이미지 텐서 변환 (HWC uint8 → CHW float32 [0, 1])
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def make_obs_batch(obs: dict, device: torch.device) -> dict:
    # 1. 관측 dict 를 정책 입력 형식으로 변환
    return {
        CAMERA_KEY: to_chw_tensor(obs["image"]).to(device),
        SIDE_CAMERA_KEY: to_chw_tensor(obs["side_image"]).to(device),
        STATE_KEY: torch.tensor(obs["state"], dtype=torch.float32, device=device),
        "task": TASK_INSTRUCTION,
    }


def load_policy(ckpt_path: Path, device: torch.device):
    # 1. 학습된 가중치 로드
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            f"먼저 `ch08/01-train-smolvla/train.py` 를 실행해 학습을 끝내주세요."
        )
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    policy.to(device)
    policy.eval()

    # 2. 학습 시 저장된 전·후처리 파이프라인 로드 (역정규화·토크나이저 재현)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def run_episode(
    env: SoArm101Env,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    device: torch.device,
) -> dict:
    # 1. 환경 초기화 + 정책 큐 리셋
    obs = env.reset()
    policy.reset()

    block_body_id = env.model.body("block").id
    gripper_qpos_idx = int(env.model.joint("gripper").qposadr[0])

    z_traj: list[float] = []
    grasp_streak = 0
    grasp_engaged = False

    # 2. 한 에피소드 추론 루프
    for _step in range(MAX_STEPS):
        # 2-1. 관측 → 정책 입력 (정규화·토크나이즈)
        batch = make_obs_batch(obs, device)
        batch = preprocessor(batch)

        # 2-2. 정책 추론 + 후처리 (역정규화)
        with torch.inference_mode():
            action_t = policy.select_action(batch)
        action_t = postprocessor(action_t)
        action = action_t.detach().cpu().numpy().reshape(-1)[:6]

        # 2-3. 그리퍼 닫힘 + 양쪽 턱 접촉 streak 만족 시 grasp 보조 활성 (수집 시 동작 미러)
        if not grasp_engaged:
            gripper_qpos = float(env.data.qpos[gripper_qpos_idx])
            if gripper_qpos <= GRASP_CLOSE_QPOS and env.block_pinched():
                grasp_streak += 1
                if grasp_streak >= GRASP_CONTACT_STREAK:
                    env.set_grasp(True)
                    grasp_engaged = True
            else:
                grasp_streak = 0

        # 2-4. 환경 적용 + 블록 z 기록
        obs, _ = env.step(action)
        z_traj.append(float(env.data.xpos[block_body_id][2]))

    # 3. 마지막 10 프레임 중 LIFT_HOLD_FRAMES 이상 임계 충족 시 성공
    final_window = z_traj[-10:]
    lifted_count = sum(1 for z in final_window if z >= LIFT_SUCCESS_Z)
    success = lifted_count >= LIFT_HOLD_FRAMES
    return {
        "success": success,
        "max_z": float(np.max(z_traj)),
        "final_z": z_traj[-1],
        "grasp_engaged": grasp_engaged,
        "z_traj": z_traj,
    }


def plot_results(results: list[dict], success_rate: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 1. 좌측 패널: 에피소드별 블록 z 궤적
    ax = axes[0]
    for r in results:
        color = "tab:green" if r["success"] else "tab:red"
        ax.plot(r["z_traj"], color=color, alpha=0.55, linewidth=1.0)
    ax.axhline(
        LIFT_SUCCESS_Z,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"threshold z={LIFT_SUCCESS_Z}",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("block z (m)")
    ax.set_title("block height trajectory per episode")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 2. 우측 패널: 에피소드별 최대 z 막대 (성공/실패 색)
    ax = axes[1]
    max_zs = [r["max_z"] for r in results]
    colors = ["tab:green" if r["success"] else "tab:red" for r in results]
    ax.bar(range(len(results)), max_zs, color=colors)
    ax.axhline(LIFT_SUCCESS_Z, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("episode")
    ax.set_ylabel("max block z (m)")
    ax.set_title(f"max block height per episode (success rate {success_rate:.0%})")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"SmolVLA pick-the-block evaluation ({len(results)} episodes)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"그래프 저장: {PLOT_PATH}")


def main() -> None:
    total_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 정책·처리 파이프라인 로드
    policy, preprocessor, postprocessor = load_policy(CKPT_PATH, device)

    # 2. 환경 생성 + 시드 고정 (동일 블록 배치 재현)
    env = SoArm101Env(control_hz=CONTROL_HZ)
    env._rng = np.random.default_rng(SEED)

    # 3. 에피소드 루프
    print(f"평가 시작: episodes={N_EPISODES}, max_steps={MAX_STEPS}, ckpt={CKPT_PATH}")
    results: list[dict] = []
    try:
        for ep in range(N_EPISODES):
            ep_start = time.perf_counter()
            result = run_episode(env, policy, preprocessor, postprocessor, device)
            ep_elapsed = time.perf_counter() - ep_start
            print(
                f"[ep {ep:02d}] success={str(result['success']):<5} "
                f"max_z={result['max_z']:+.3f} final_z={result['final_z']:+.3f} "
                f"grasp={'Y' if result['grasp_engaged'] else 'N'} "
                f"({ep_elapsed:.1f}s)"
            )
            results.append(result)
    finally:
        env.close()

    # 4. 통계 집계
    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / len(results) if results else 0.0
    avg_max_z = float(np.mean([r["max_z"] for r in results])) if results else 0.0
    n_grasp = sum(1 for r in results if r["grasp_engaged"])

    elapsed = time.perf_counter() - total_start
    print("---")
    print(f"성공률: {n_success}/{len(results)} = {success_rate:.1%}")
    print(f"grasp 보조 활성 에피소드: {n_grasp}/{len(results)}")
    print(f"평균 최대 z: {avg_max_z:.3f}m  (임계 {LIFT_SUCCESS_Z}m)")
    print(f"총 소요: {elapsed:.1f}s")

    # 5. 요약 JSON 저장 (z 궤적은 그래프 외 보존 불필요)
    summary = {
        "n_episodes": len(results),
        "n_success": n_success,
        "success_rate": success_rate,
        "avg_max_z": avg_max_z,
        "lift_success_z": LIFT_SUCCESS_Z,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "ckpt": str(CKPT_PATH),
        "episodes": [
            {
                "success": r["success"],
                "max_z": r["max_z"],
                "final_z": r["final_z"],
                "grasp_engaged": r["grasp_engaged"],
            }
            for r in results
        ],
    }
    RESULT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"요약 저장: {RESULT_JSON}")

    # 6. 그래프 출력
    plot_results(results, success_rate)


if __name__ == "__main__":
    main()
