#!/usr/bin/env python3

"""aug200 5단계(4k/8k/12k/16k/20k) 체크포인트 평가 — ch08 5단계 대응."""

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

CH06_DIR = Path(__file__).resolve().parents[2] / "ch06" / "01-lerobot-record"
sys.path.insert(0, str(CH06_DIR))

from env import SoArm101Env  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT_ROOT = HERE / "checkpoints" / "soarm-aug200-20k"

TASK_INSTRUCTION = "pick the block"
N_EPISODES = 20
MAX_STEPS = 200
CONTROL_HZ = 20
SEED = 42

LIFT_SUCCESS_Z = 0.12
LIFT_HOLD_FRAMES = 5
GRASP_CLOSE_QPOS = 0.9
GRASP_CONTACT_STREAK = 3

# 1. ch08 1k/2k/3k/4k/5k 의 4×
EVAL_STEPS = [4000, 8000, 12000, 16000, 20000]

OUT_JSON = HERE / "eval.json"
OUT_PNG = HERE / "images" / "eval.png"

CAMERA_KEY = "observation.images.top"
SIDE_CAMERA_KEY = "observation.images.side"
STATE_KEY = "observation.state"


def to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def make_obs_batch(obs: dict, device: torch.device) -> dict:
    return {
        CAMERA_KEY: to_chw_tensor(obs["image"]).to(device),
        SIDE_CAMERA_KEY: to_chw_tensor(obs["side_image"]).to(device),
        STATE_KEY: torch.tensor(obs["state"], dtype=torch.float32, device=device),
        "task": TASK_INSTRUCTION,
    }


def load_policy(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            "먼저 `uv run python train.py` 를 실행하세요."
        )
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    policy.to(device)
    policy.eval()
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
    obs = env.reset()
    policy.reset()

    block_body_id = env.model.body("block").id
    gripper_qpos_idx = int(env.model.joint("gripper").qposadr[0])

    z_traj: list[float] = []
    grasp_streak = 0
    grasp_engaged = False

    for _step in range(MAX_STEPS):
        batch = make_obs_batch(obs, device)
        batch = preprocessor(batch)

        with torch.inference_mode():
            action_t = policy.select_action(batch)
        action_t = postprocessor(action_t)
        action = action_t.detach().cpu().numpy().reshape(-1)[:6]

        if not grasp_engaged:
            gripper_qpos = float(env.data.qpos[gripper_qpos_idx])
            if gripper_qpos <= GRASP_CLOSE_QPOS and env.block_pinched():
                grasp_streak += 1
                if grasp_streak >= GRASP_CONTACT_STREAK:
                    env.set_grasp(True)
                    grasp_engaged = True
            else:
                grasp_streak = 0

        obs, _ = env.step(action)
        z_traj.append(float(env.data.xpos[block_body_id][2]))

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


def ckpt_path(step: int) -> Path:
    return CKPT_ROOT / f"step_{step}"


def evaluate_one(ckpt: Path, device: torch.device) -> dict:
    policy, preprocessor, postprocessor = load_policy(ckpt, device)
    env = SoArm101Env(control_hz=CONTROL_HZ)
    env._rng = np.random.default_rng(SEED)

    results: list[dict] = []
    try:
        for _ep in range(N_EPISODES):
            results.append(
                run_episode(env, policy, preprocessor, postprocessor, device)
            )
    finally:
        env.close()

    n_success = sum(1 for r in results if r["success"])
    n_grasp = sum(1 for r in results if r["grasp_engaged"])
    return {
        "train_step": int(ckpt.name.removeprefix("step_")),
        "ckpt": str(ckpt),
        "n_success": n_success,
        "success_rate": n_success / len(results),
        "avg_max_z": float(np.mean([r["max_z"] for r in results])),
        "n_grasp": n_grasp,
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


def plot_stages(rows: list[dict]) -> None:
    steps = [r["train_step"] for r in rows]
    rates = [r["success_rate"] * 100 for r in rows]
    avgs = [r["avg_max_z"] for r in rows]
    grasps = [r["n_grasp"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    ax.plot(steps, rates, marker="o", color="tab:blue")
    ax.set_xlabel("train step")
    ax.set_ylabel("success rate (%)")
    ax.set_title("aug200 5-stage (20 ep, seed=42)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, avgs, marker="o", color="tab:green")
    ax.axhline(LIFT_SUCCESS_Z, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("train step")
    ax.set_ylabel("avg max block z (m)")
    ax.set_title("average max z")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(steps, grasps, marker="o", color="tab:orange")
    ax.set_xlabel("train step")
    ax.set_ylabel("grasp engaged (count)")
    ax.set_title(f"grasp activations (out of {N_EPISODES})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"그래프 저장: {OUT_PNG}")


def main() -> None:
    if not CKPT_ROOT.exists():
        raise FileNotFoundError(
            f"체크포인트 없음: {CKPT_ROOT}\n"
            "먼저 `uv run python train.py` (20k step) 를 실행하세요."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    total_start = time.perf_counter()

    for i, step in enumerate(EVAL_STEPS, 1):
        ckpt = ckpt_path(step)
        if not ckpt.exists():
            print(f"건너뜀: {ckpt} 없음")
            continue
        t0 = time.perf_counter()
        row = evaluate_one(ckpt, device)
        elapsed = time.perf_counter() - t0
        rows.append(row)
        print(
            f"[{i}/{len(EVAL_STEPS)}] step_{step:<5} "
            f"success={row['success_rate']:.1%} ({row['n_success']}/20) "
            f"avg_z={row['avg_max_z']:.3f} grasp={row['n_grasp']}/20 "
            f"({elapsed:.0f}s)"
        )

    if not rows:
        print("평가할 체크포인트가 없습니다.")
        return

    best = max(rows, key=lambda r: r["success_rate"])
    summary = {
        "dataset": "local/so_arm101_block_picking_aug200",
        "eval_steps": EVAL_STEPS,
        "ch08_equivalent_steps": [s // 4 for s in EVAL_STEPS],
        "seed": SEED,
        "n_episodes": N_EPISODES,
        "best": best,
        "stages": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print("---")
    print(f"best: step_{best['train_step']} success={best['success_rate']:.1%}")
    print(f"저장: {OUT_JSON}")
    print(f"총 소요: {time.perf_counter() - total_start:.0f}s")

    plot_stages(rows)


if __name__ == "__main__":
    main()
