#!/usr/bin/env python3

"""SmolVLA 5,000 step 파인튜닝 예제 (ch07/01 베이스)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors,
)
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader

# 1. 데이터셋·체크포인트 식별자
DATASET_REPO_ID = "makepluscode/so_arm101_block_picking_main"
BASE_CHECKPOINT = "lerobot/smolvla_base"

# 2. 학습 하이퍼파라미터 (ch07/01 대비 학습량 25배 확대)
LR = 1e-4
BATCH_SIZE = 4
TOTAL_STEPS = 5000  # ch07/01: 200 → 5000 (데이터셋 약 2.1 epoch)
WARMUP_STEPS = 500  # ch07/01: 20 → 500 (TOTAL_STEPS 의 10%)
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
LOG_EVERY = 100  # ch07/01: 10 → 100 (로그 1/10 축소)
CKPT_EVERY = 1000  # ch07/01: 100 → 1000 (중간 ckpt 4개 + 최종)

# 3. 결과 저장 경로 (ch07/01 와 분리, ch08/02 평가에서 같은 경로 참조)
OUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "soarm-5000-run1"


def build_dataset() -> LeRobotDataset:
    # 1. 액션 청크 윈도우 정의 (chunk_size 만큼 미래 액션 수집)
    cfg = SmolVLAConfig()
    fps = 20
    delta_timestamps = {
        "action": [i / fps for i in range(cfg.chunk_size)],
    }

    # 2. Hugging Face Hub 데이터셋 로드
    print(f"Hub 데이터셋 사용: {DATASET_REPO_ID}")
    return LeRobotDataset(DATASET_REPO_ID, delta_timestamps=delta_timestamps)


def build_policy(dataset: LeRobotDataset, device: torch.device) -> SmolVLAPolicy:
    # 1. 데이터셋 feature 를 정책 입력·출력 형식으로 변환
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    input_features = {k: v for k, v in features.items() if k not in output_features}

    # 2. config 에 데이터셋 키(top/side)와 디바이스 주입
    cfg = SmolVLAConfig()
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.device = str(device)

    # 3. 사전학습 SmolVLA 로드 + 데이터셋 통계 주입
    policy = SmolVLAPolicy.from_pretrained(
        BASE_CHECKPOINT,
        config=cfg,
        dataset_stats=dataset.meta.stats,
    )
    policy.train()
    policy.to(device)
    return policy


def build_processors(policy: SmolVLAPolicy, dataset: LeRobotDataset):
    # 1. 정규화·토크나이즈·디바이스 이동을 묶은 전·후처리 파이프라인 생성 (device 는 policy.config 에 이미 주입됨)
    return make_smolvla_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset.meta.stats,
    )


def cosine_with_warmup(step: int) -> float:
    # 1. warmup 구간은 선형 증가
    if step < WARMUP_STEPS:
        return float(step) / float(max(1, WARMUP_STEPS))
    # 2. 이후 구간은 코사인 감쇠
    progress = (step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    out_dir: Path,
    step: int | str,
) -> Path:
    # 1. 체크포인트 디렉터리 생성
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. 모델 가중치·config 저장
    policy.save_pretrained(ckpt_dir)

    # 3. 전·후처리 파이프라인 저장 (역정규화·토크나이저 재현용)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    return ckpt_dir


def main() -> None:
    total_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 데이터셋 로드
    dataset = build_dataset()
    print(f"데이터셋: 에피소드 {dataset.num_episodes}개, 프레임 {len(dataset)}개")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # 2. 사전학습 정책 + 부분 파인튜닝 (config 기본값: vision freeze + expert only)
    policy = build_policy(dataset, device)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"학습 대상 파라미터: {trainable / 1e6:.1f}M / 전체 {total / 1e6:.1f}M")

    # 3. 입력·출력 처리 파이프라인 구성
    preprocessor, postprocessor = build_processors(policy, dataset)

    # 4. AdamW + Cosine Warmup 스케줄러 구성
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_with_warmup
    )

    # 5. 학습 루프
    print(f"학습 시작: total_steps={TOTAL_STEPS}, batch={BATCH_SIZE}, lr={LR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step = 0
    loader_iter = iter(loader)
    while step < TOTAL_STEPS:
        # 5-1. 배치 추출 (loader 소진 시 재생성)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # 5-2. 전처리 (정규화·토크나이즈·디바이스 이동)
        batch = preprocessor(batch)

        # 5-3. forward + loss (Flow Matching, 정규화된 액션 공간)
        loss, _ = policy.forward(batch)

        # 5-4. 역전파·grad clip·옵티마이저 step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()
        scheduler.step()

        # 5-5. 로그 출력
        if step % LOG_EVERY == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[step {step:04d}] loss={loss.item():.4f} lr={lr_now:.2e}")

        # 5-6. 중간 체크포인트 저장
        if step > 0 and step % CKPT_EVERY == 0:
            ckpt_dir = save_checkpoint(
                policy, preprocessor, postprocessor, OUT_DIR, step
            )
            print(f"체크포인트 저장: {ckpt_dir}")

        step += 1

    # 6. 최종 체크포인트 저장
    final_dir = save_checkpoint(policy, preprocessor, postprocessor, OUT_DIR, "final")
    print(f"최종 체크포인트: {final_dir}")

    elapsed = time.perf_counter() - total_start
    print(f"총 학습 시간: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
