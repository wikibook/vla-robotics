"""ch05 PushT 평가 루프 검증 (main import — main()·Hub 미호출)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from main import (
    DATASET_REPO,
    GIF_STRIDE,
    MAX_STEPS,
    MODEL_REPO,
    NUM_EPISODES,
    SNAP_DIR,
    run_episode,
)


def _pixels(val: int) -> np.ndarray:
    # 1. 가짜 RGB 관측
    return np.full((4, 4, 3), val, dtype=np.uint8)


def _make_mocks(
    post_action: np.ndarray,
    rewards: list[float],
    *,
    terminated_at: int | None = None,
    truncated_at: int | None = None,
):
    # 1. env·policy·processor 목
    step_idx = {"n": 0}

    def reset(seed=None):
        step_idx["n"] = 0
        return {"pixels": _pixels(1)}, {}

    def step(action):
        i = step_idx["n"]
        step_idx["n"] += 1
        reward = rewards[i] if i < len(rewards) else 0.0
        terminated = terminated_at is not None and step_idx["n"] >= terminated_at
        truncated = truncated_at is not None and step_idx["n"] >= truncated_at
        return {"pixels": _pixels(int(reward * 100))}, reward, terminated, truncated, {}

    env = MagicMock()
    env.reset.side_effect = reset
    env.step.side_effect = step

    policy = MagicMock()
    policy.select_action.return_value = {}

    preprocessor = MagicMock(side_effect=lambda obs: obs)
    gym_obs = MagicMock()
    gym_obs.observation.side_effect = lambda obs: obs

    post_tensor = MagicMock()
    post_tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = post_action
    postprocessor = MagicMock(return_value=post_tensor)

    return env, policy, preprocessor, postprocessor, gym_obs


def test_eval_config_constants():
    # 1. 평가 설정 상수
    assert MAX_STEPS == 300
    assert NUM_EPISODES == 10
    assert GIF_STRIDE == 10
    assert MODEL_REPO == "lerobot/diffusion_pusht"
    assert DATASET_REPO == "lerobot/pusht"
    assert SNAP_DIR.name == "snapshots"


def test_run_episode_collects_frames_on_stride(monkeypatch):
    # 1. GIF_STRIDE 간격 프레임 수집
    monkeypatch.setattr("main.MAX_STEPS", 25)
    env, policy, pre, post, gym_obs = _make_mocks(
        np.array([0.0, 0.0]),
        rewards=[0.1] * 25,
    )
    max_reward, frames, _ = run_episode(env, policy, pre, post, gym_obs, seed=0)
    assert max_reward == pytest.approx(0.1)
    assert len(frames) == 3
    policy.reset.assert_called_once()


def test_run_episode_tracks_max_reward_frame():
    # 1. 최대 보상 시점 프레임 갱신
    env, policy, pre, post, gym_obs = _make_mocks(
        np.array([0.0, 0.0]),
        rewards=[0.2, 0.9, 0.3],
        terminated_at=4,
    )
    _, _, max_frame = run_episode(env, policy, pre, post, gym_obs, seed=1)
    assert int(max_frame[0, 0, 0]) == 90


def test_run_episode_squeezes_batch_action_dim():
    # 1. batch 차원 (1, D) → (D,) 정리
    env, policy, pre, post, gym_obs = _make_mocks(
        np.array([[0.5, -0.5]]),
        rewards=[0.0],
        terminated_at=2,
    )
    run_episode(env, policy, pre, post, gym_obs, seed=2)
    passed = env.step.call_args[0][0]
    assert passed.shape == (2,)
    assert passed[0] == pytest.approx(0.5)


def test_run_episode_stops_on_terminated():
    # 1. terminated 시 조기 종료
    env, policy, pre, post, gym_obs = _make_mocks(
        np.array([0.0, 0.0]),
        rewards=[0.5] * 50,
        terminated_at=3,
    )
    run_episode(env, policy, pre, post, gym_obs, seed=3)
    assert env.step.call_count == 3


def test_run_episode_stops_on_truncated():
    # 1. truncated 시 조기 종료
    env, policy, pre, post, gym_obs = _make_mocks(
        np.array([0.0, 0.0]),
        rewards=[0.4] * 50,
        truncated_at=2,
    )
    max_reward, _, _ = run_episode(env, policy, pre, post, gym_obs, seed=4)
    assert max_reward == pytest.approx(0.4)
    assert env.step.call_count == 2


def test_run_episode_inference_mode_used():
    # 1. select_action·전후처리 호출
    env, policy, pre, post, gym_obs = _make_mocks(
        np.array([0.0, 0.0]),
        rewards=[0.0],
        terminated_at=2,
    )
    run_episode(env, policy, pre, post, gym_obs, seed=5)
    assert policy.select_action.call_count == 2
    assert pre.call_count == 2
    assert post.call_count == 2


def test_gym_pusht_env_reset():
    # 1. PushT 환경 등록·reset
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=0)
    assert "pixels" in obs
    assert "agent_pos" in obs
    assert obs["pixels"].dtype == np.uint8
    env.close()
