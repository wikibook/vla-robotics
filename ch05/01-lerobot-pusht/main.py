"""PushT Diffusion Policy 평가 예제."""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import VanillaObservationProcessorStep
from PIL import Image

MODEL_REPO = "lerobot/diffusion_pusht"
DATASET_REPO = "lerobot/pusht"
# 1. 평가 설정
MAX_STEPS = 300
NUM_EPISODES = 10
GIF_STRIDE = 10
SNAP_DIR = Path(__file__).resolve().parent / "snapshots"


def build_processors(policy, dataset, device):
    """학습 통계 기반 전·후처리기 생성.

    preprocessor: 관측을 데이터셋 통계로 정규화하고 정책 실행 장치로 이동
    postprocessor: 정책 출력(정규화된 action)을 환경이 받는 실제 좌표계 스케일로 복원
    """
    overrides = {"device_processor": {"device": str(device)}}
    return make_pre_post_processors(
        policy.config,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=overrides,
    )


def run_episode(env, policy, preprocessor, postprocessor, gym_obs_processor, seed):
    """에피소드 1회 실행 — LeRobot 추론 파이프라인의 핵심 루프."""

    # 1. 에피소드 상태 초기화
    policy.reset()
    obs, _ = env.reset(seed=seed)
    max_reward = 0.0
    max_reward_frame = np.asarray(obs["pixels"], dtype=np.uint8)
    frames = []

    for step in range(MAX_STEPS):
        if step % GIF_STRIDE == 0:
            frames.append(np.asarray(obs["pixels"], dtype=np.uint8))

        # 2. 관측 키 변환
        policy_obs = gym_obs_processor.observation(obs)

        # 3. 관측 전처리
        policy_input = preprocessor(policy_obs)

        # 4. 정책 추론
        with torch.inference_mode():
            action = policy.select_action(policy_input)

        # 5. 행동 후처리
        action = postprocessor(action).detach().float().cpu().numpy()

        # 6. batch 차원 정리
        if action.ndim == 2:
            action = action[0]

        # 7. 환경 제어
        obs, reward, terminated, truncated, _ = env.step(action)
        if float(reward) > max_reward:
            max_reward = float(reward)
            # 7-1. 최대 보상 프레임 갱신
            max_reward_frame = np.asarray(obs["pixels"], dtype=np.uint8)

        if terminated or truncated:
            break

    return max_reward, frames, max_reward_frame


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device.type}")

    # 2. 데이터셋 로드
    dataset = LeRobotDataset(DATASET_REPO)
    print(f"데이터셋: 에피소드 {dataset.num_episodes}개, 프레임 {len(dataset)}개")

    # 3. 사전학습 정책 로드
    policy = DiffusionPolicy.from_pretrained(MODEL_REPO)
    policy.eval()
    policy.to(device)

    # 4. 전처리·후처리기 생성
    preprocessor, postprocessor = build_processors(policy, dataset, device)
    gym_obs_processor = VanillaObservationProcessorStep()

    # 5. PushT 환경 생성
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    rewards = []

    # 6. 에피소드 반복 평가
    for i in range(NUM_EPISODES):
        max_reward, frames, max_reward_frame = run_episode(
            env,
            policy,
            preprocessor,
            postprocessor,
            gym_obs_processor,
            seed=i,
        )
        rewards.append(max_reward)

        # 6-1. GIF·PNG 저장
        if frames:
            imageio.mimsave(SNAP_DIR / f"pusht_ep{i + 1:02d}.gif", frames, fps=10)
        Image.fromarray(max_reward_frame).save(
            SNAP_DIR / f"pusht_ep{i + 1:02d}_max_reward.png"
        )

        mark = "✅" if max_reward >= 0.90 else "❌"
        print(
            f"  에피소드 {i + 1:2d}/{NUM_EPISODES}: 최대보상 {max_reward:.2f}  {mark}"
        )

    env.close()

    # 7. 최종 성능 요약
    n_success = sum(1 for r in rewards if r >= 0.90)
    avg_reward = sum(rewards) / len(rewards)
    print(f"\n결과: {n_success}/{NUM_EPISODES} 성공, 평균보상 {avg_reward:.2f}")
    print(f"스냅샷: {SNAP_DIR}")


if __name__ == "__main__":
    main()
