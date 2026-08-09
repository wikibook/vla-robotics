#!/usr/bin/env python3

"""SO-ARM 101 시연 데이터 수집 예제."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

HF_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_ROOT / "datasets")
os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from env import SoArm101Env  # noqa: E402
from scripted import EPISODE_STEP_BUDGET, ScriptedPickPolicy  # noqa: E402

DATASET_REPO_ID = "local/so_arm101_block_picking_main"
DATASET_ROOT = Path("./datasets")
TASK_LABEL = "pick the block"
DEFAULT_NUM_EPISODES = int(os.environ.get("SOARM101_NUM_EPISODES", "1"))
CONTROL_HZ = 20
MAX_STEPS = EPISODE_STEP_BUDGET

CAMERA_KEY = "observation.images.top"
SIDE_CAMERA_KEY = "observation.images.side"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

FEATURES = {
    CAMERA_KEY: {
        "dtype": "video",
        "shape": (3, 128, 128),
        "names": ["channel", "height", "width"],
    },
    SIDE_CAMERA_KEY: {
        "dtype": "video",
        "shape": (3, 256, 256),
        "names": ["channel", "height", "width"],
    },
    STATE_KEY: {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
    ACTION_KEY: {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
}


@dataclass
class CollectionRuntime:
    dataset: LeRobotDataset
    env: SoArm101Env
    policy: ScriptedPickPolicy
    control_dt: float
    num_episodes: int


def configure_hf_cache() -> None:
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (HF_CACHE_ROOT / "datasets").mkdir(parents=True, exist_ok=True)


def load_or_create_dataset(dataset_root: Path) -> LeRobotDataset:
    if dataset_root.exists():
        return LeRobotDataset.resume(DATASET_REPO_ID, root=dataset_root)
    return LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=CONTROL_HZ,
        root=dataset_root,
        features=FEATURES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO-ARM 101 시연 데이터 수집")
    parser.add_argument(
        "num_episodes",
        nargs="?",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="저장할 에피소드 수",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def make_frame(obs: dict[str, np.ndarray], action: np.ndarray) -> dict[str, Any]:
    return {
        CAMERA_KEY: torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 255.0,
        SIDE_CAMERA_KEY: torch.from_numpy(obs["side_image"]).permute(2, 0, 1).float()
        / 255.0,
        STATE_KEY: torch.tensor(obs["state"], dtype=torch.float32),
        ACTION_KEY: torch.tensor(action, dtype=torch.float32),
        "task": TASK_LABEL,
    }


def collect_episode_frames(runtime: CollectionRuntime) -> list[dict[str, Any]]:
    obs = runtime.env.reset()
    runtime.policy.reset_episode()
    frames: list[dict[str, Any]] = []

    for _ in range(MAX_STEPS):
        loop_start = time.perf_counter()
        action = np.asarray(runtime.policy.get_action(), dtype=np.float32)
        next_obs, _ = runtime.env.step(action)
        frames.append(make_frame(obs, action))
        obs = next_obs

        if runtime.policy.save_requested:
            return frames

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, runtime.control_dt - elapsed))

    return []


def save_episode(dataset: LeRobotDataset, frames: list[dict[str, Any]]) -> int:
    episode_index = dataset.meta.total_episodes
    for frame in frames:
        dataset.add_frame(frame)
    dataset.save_episode()
    return episode_index


def print_banner(dataset_path: Path, num_episodes: int) -> None:
    estimated_episode_seconds = MAX_STEPS / CONTROL_HZ
    estimated_total_seconds = estimated_episode_seconds * num_episodes
    print(
        f"\n[01-lerobot-record] scripted pick 데이터 수집\n"
        f"  목표 에피소드  : {num_episodes}\n"
        f"  step 한도      : {MAX_STEPS}\n"
        f"  예상 소요 시간 : {format_duration(estimated_total_seconds)} "
        f"(에피소드당 {format_duration(estimated_episode_seconds)})\n"
        f"  데이터셋 경로  : {dataset_path}\n"
    )


def print_saved_episode(
    episode_1based: int,
    num_episodes: int,
    frame_count: int,
    episode_index: int,
    result: str,
) -> None:
    print(
        f"\n에피소드 {episode_1based}/{num_episodes} 저장\n"
        f"  프레임 수      : {frame_count}\n"
        f"  에피소드 인덱스: {episode_index}\n"
        f"  결과           : {result}"
    )


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_path = DATASET_ROOT / DATASET_REPO_ID
    runtime: CollectionRuntime | None = None

    try:
        # 1. 로컬 캐시 디렉터리 준비
        configure_hf_cache()

        # 2. 데이터셋·환경·정책 준비
        dataset = load_or_create_dataset(dataset_path)
        env = SoArm101Env(control_hz=CONTROL_HZ)
        policy = ScriptedPickPolicy(env)
        runtime = CollectionRuntime(
            dataset=dataset,
            env=env,
            policy=policy,
            control_dt=1.0 / CONTROL_HZ,
            num_episodes=args.num_episodes,
        )

        # 3. 수집 시작 안내
        print_banner(dataset_path, runtime.num_episodes)

        # 4. 시연 수집 반복
        saved = 0
        while saved < runtime.num_episodes:
            # 4-1. 한 에피소드 frame 수집
            frames = collect_episode_frames(runtime)
            if not frames:
                print(f"\n에피소드 {saved + 1}/{runtime.num_episodes} 실패, 재시도")
                continue

            # 4-2. 성공 에피소드 저장
            episode_index = save_episode(runtime.dataset, frames)
            saved += 1
            print_saved_episode(
                episode_1based=saved,
                num_episodes=runtime.num_episodes,
                frame_count=len(frames),
                episode_index=episode_index,
                result=runtime.policy.last_grasp_cause,
            )

        # 5. 수집 결과 요약
        print(f"\n총 저장 에피소드: {saved}")
        print(f"데이터셋 경로: {dataset_path}")
    finally:
        if runtime is not None:
            # 6. MuJoCo 리소스 정리
            runtime.env.close()
        total_elapsed = time.perf_counter() - total_start
        print(
            f"\n총 실행 시간: {format_duration(total_elapsed)} ({total_elapsed:.2f}s)"
        )


if __name__ == "__main__":
    main()
