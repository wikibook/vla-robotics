#!/usr/bin/env python3

"""SmolVLA 첫 추론 예제."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

# 1. ch06 SoArm101Env 재사용 경로 등록
CH06_DIR = Path(__file__).resolve().parents[2] / "ch06" / "01-lerobot-record"
sys.path.insert(0, str(CH06_DIR))

from env import SoArm101Env  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

# 2. 추론 설정
CKPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "01-train-smolvla/checkpoints/soarm-200-run1/step_final"
)
TASK_INSTRUCTION = "pick the block"
MAX_STEPS = 200
CONTROL_HZ = 20

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
            f"먼저 `ch07/01-train-smolvla/train.py` 를 실행해 학습을 끝내주세요."
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


def main() -> None:
    total_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 정책·처리 파이프라인 로드
    policy, preprocessor, postprocessor = load_policy(CKPT_PATH, device)
    policy.reset()

    # 2. MuJoCo 환경 생성 + 초기 관측
    env = SoArm101Env(control_hz=CONTROL_HZ)
    obs = env.reset()
    control_dt = 1.0 / CONTROL_HZ

    # 3. 추론 루프
    print(f"추론 시작: max_steps={MAX_STEPS}, ckpt={CKPT_PATH}")
    try:
        for step in range(MAX_STEPS):
            loop_start = time.perf_counter()

            # 3-1. 관측 → 정책 입력 (정규화·토크나이즈)
            batch = make_obs_batch(obs, device)
            batch = preprocessor(batch)

            # 3-2. 정책 추론 (내부 큐가 액션 청크 관리)
            with torch.inference_mode():
                action_t = policy.select_action(batch)

            # 3-3. 후처리 (역정규화 + cpu 이동) → 6 DOF 액션 추출
            action_t = postprocessor(action_t)
            action = action_t.detach().cpu().numpy().reshape(-1)[:6]

            # 3-4. 환경 적용
            obs, _ = env.step(action)

            # 3-5. 10 스텝마다 상태 출력
            if step % 10 == 0:
                qpos = obs["state"]
                print(
                    f"[t={step:03d}] action=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f},"
                    f"{action[3]:+.2f},{action[4]:+.2f},{action[5]:+.2f}] "
                    f"qpos[0:3]=[{qpos[0]:+.2f},{qpos[1]:+.2f},{qpos[2]:+.2f}]"
                )

            # 3-6. 제어 주기 보정
            elapsed = time.perf_counter() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    finally:
        env.close()

    elapsed = time.perf_counter() - total_start
    print(f"추론 종료: 총 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
