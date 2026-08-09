#!/usr/bin/env python3

"""SmolVLA 20,000 step 파인튜닝 — offline 증강 200ep (ch08 5k × 4)."""

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

# 1. offline 증강 200ep 로컬 데이터셋 (01-data-augment 출력)
DATASET_REPO_ID = "local/so_arm101_block_picking_aug200"
AUGMENT_DIR = Path(__file__).resolve().parent.parent / "01-data-augment"
DATASET_ROOT = AUGMENT_DIR / "datasets" / "local/so_arm101_block_picking_aug200"
BASE_CHECKPOINT = "lerobot/smolvla_base"

# 2. ch08 1k/2k/3k/4k/5k 의 4× → 4k/8k/12k/16k/20k
LR = 1e-4
BATCH_SIZE = 4
TOTAL_STEPS = 20000
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
LOG_EVERY = 100
CKPT_EVERY = 4000

# 3. 체크포인트 저장 경로
OUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "soarm-aug200-20k"


def build_dataset() -> LeRobotDataset:
    cfg = SmolVLAConfig()
    fps = 20
    delta_timestamps = {
        "action": [i / fps for i in range(cfg.chunk_size)],
    }

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"증강 데이터셋이 없습니다: {DATASET_ROOT}\n"
            "먼저 `cd ../01-data-augment && uv run python augment_dataset.py` 를 실행하세요."
        )

    print(f"로컬 증강 데이터셋 사용: {DATASET_REPO_ID}")
    return LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )


def build_policy(dataset: LeRobotDataset, device: torch.device) -> SmolVLAPolicy:
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    input_features = {k: v for k, v in features.items() if k not in output_features}

    cfg = SmolVLAConfig()
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.device = str(device)

    policy = SmolVLAPolicy.from_pretrained(
        BASE_CHECKPOINT,
        config=cfg,
        dataset_stats=dataset.meta.stats,
    )
    policy.train()
    policy.to(device)
    return policy


def build_processors(policy: SmolVLAPolicy, dataset: LeRobotDataset):
    return make_smolvla_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset.meta.stats,
    )


def cosine_with_warmup(step: int) -> float:
    if step < WARMUP_STEPS:
        return float(step) / float(max(1, WARMUP_STEPS))
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
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    return ckpt_dir


def main() -> None:
    total_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    policy = build_policy(dataset, device)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"학습 대상 파라미터: {trainable / 1e6:.1f}M / 전체 {total / 1e6:.1f}M")

    preprocessor, postprocessor = build_processors(policy, dataset)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_with_warmup
    )

    print(
        f"학습 시작: total_steps={TOTAL_STEPS}, ckpt_every={CKPT_EVERY}, "
        f"batch={BATCH_SIZE}, lr={LR}"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step = 0
    loader_iter = iter(loader)
    while step < TOTAL_STEPS:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[step {step:05d}] loss={loss.item():.4f} lr={lr_now:.2e}")

        if step > 0 and step % CKPT_EVERY == 0:
            ckpt_dir = save_checkpoint(
                policy, preprocessor, postprocessor, OUT_DIR, step
            )
            print(f"체크포인트 저장: {ckpt_dir}")

        step += 1

    final_dir = save_checkpoint(
        policy, preprocessor, postprocessor, OUT_DIR, TOTAL_STEPS
    )
    print(f"최종 체크포인트: {final_dir}")

    elapsed = time.perf_counter() - total_start
    print(f"총 학습 시간: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
