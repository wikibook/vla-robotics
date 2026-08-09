#!/usr/bin/env python3

"""Hub 50ep 데이터셋을 offline 증강해 200ep 로 확장."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms import v2

# 1. 입·출력 데이터셋 식별자
SOURCE_REPO_ID = "makepluscode/so_arm101_block_picking_main"
OUTPUT_REPO_ID = "local/so_arm101_block_picking_aug200"
OUTPUT_ROOT = Path(__file__).resolve().parent / "datasets" / OUTPUT_REPO_ID

# 2. 카메라 키 (LeRobotDataset feature 이름과 동일)
TOP_CAMERA_KEY = "observation.images.top"
SIDE_CAMERA_KEY = "observation.images.side"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEYS = (TOP_CAMERA_KEY, SIDE_CAMERA_KEY)

# 3. offline 증강 (원본 1 + 사본 AUG_COPIES = 4배 → 50→200 ep)
AUG_COPIES = 3
COLOR_JITTER_BRIGHTNESS = 0.05
COLOR_JITTER_CONTRAST = 0.05
COLOR_JITTER_SATURATION = 0.05
AUG_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hub 50ep → offline 증강 200ep")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="처리할 원본 에피소드 상한 (시험용)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=AUG_COPIES,
        help="원본당 증강 사본 수 (기본 3 → 50×4=200 ep)",
    )
    parser.add_argument(
        "--skip-augmented",
        action="store_true",
        help="원본 복사만 (디버그용)",
    )
    return parser.parse_args()


def build_image_aug(seed: int) -> v2.Compose:
    # 1. ColorJitter만 사용 (GaussianBlur는 online 실험에서 성능 하락)
    torch.manual_seed(seed)
    return v2.Compose(
        [
            v2.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
            ),
        ]
    )


def frame_from_item(item: dict) -> dict:
    # 1. add_frame 에 필요한 feature + task 만 추림
    return {
        TOP_CAMERA_KEY: item[TOP_CAMERA_KEY],
        SIDE_CAMERA_KEY: item[SIDE_CAMERA_KEY],
        STATE_KEY: item[STATE_KEY],
        ACTION_KEY: item[ACTION_KEY],
        "task": item["task"],
    }


def apply_image_aug(frame: dict, aug: v2.Compose) -> dict:
    # 1. top·side 카메라에만 증강 (액션·상태는 원본 유지)
    out = dict(frame)
    for key in IMAGE_KEYS:
        img = frame[key]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        out[key] = aug(img)
    return out


def copy_episode(source: LeRobotDataset, out: LeRobotDataset, ep_index: int) -> int:
    ep_meta = source.meta.episodes[ep_index]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])

    for idx in range(start, end):
        out.add_frame(frame_from_item(source[idx]))
    out.save_episode()
    return end - start


def copy_episode_augmented(
    source: LeRobotDataset,
    out: LeRobotDataset,
    ep_index: int,
    aug_seed: int,
) -> int:
    aug = build_image_aug(aug_seed)
    ep_meta = source.meta.episodes[ep_index]
    start = int(ep_meta["dataset_from_index"])
    end = int(ep_meta["dataset_to_index"])

    for idx in range(start, end):
        frame = frame_from_item(source[idx])
        out.add_frame(apply_image_aug(frame, aug))
    out.save_episode()
    return end - start


def create_output_dataset(source: LeRobotDataset) -> LeRobotDataset:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    return LeRobotDataset.create(
        repo_id=OUTPUT_REPO_ID,
        fps=source.meta.fps,
        features=source.meta.features,
        root=OUTPUT_ROOT,
        robot_type=source.meta.robot_type,
        batch_encoding_size=1,
    )


def verify_output(
    source: LeRobotDataset,
    out: LeRobotDataset,
    n_source: int | None,
    aug_copies: int,
    skip_augmented: bool,
) -> None:
    base = n_source if n_source else source.num_episodes
    if skip_augmented:
        expected_eps = base
    else:
        expected_eps = base * (1 + aug_copies)
    if out.num_episodes != expected_eps:
        raise RuntimeError(
            f"에피소드 수 불일치: 기대 {expected_eps}, 실제 {out.num_episodes}"
        )
    if set(out.meta.features.keys()) != set(source.meta.features.keys()):
        raise RuntimeError("feature schema 불일치")
    print(
        f"검증 통과: {out.num_episodes} ep, {len(out)} frames, "
        f"features={list(out.meta.features.keys())}"
    )


def main() -> None:
    args = parse_args()
    n_source = args.max_episodes

    print(f"원본 로드: {SOURCE_REPO_ID}")
    source = LeRobotDataset(SOURCE_REPO_ID)
    total = n_source if n_source is not None else source.num_episodes
    total = min(total, source.num_episodes)
    multiplier = 1 if args.skip_augmented else 1 + args.copies
    print(
        f"처리 대상: {total}/{source.num_episodes} ep "
        f"→ 출력 {total * multiplier} ep (copies={args.copies})"
    )

    out = create_output_dataset(source)

    for ep_index in range(total):
        n_frames = copy_episode(source, out, ep_index)
        print(f"[{ep_index + 1}/{total}] 원본 ep{ep_index} → {n_frames} frames")

        if not args.skip_augmented:
            for copy_idx in range(args.copies):
                seed = AUG_SEED + ep_index + copy_idx * 1000
                n_aug = copy_episode_augmented(source, out, ep_index, aug_seed=seed)
                print(
                    f"[{ep_index + 1}/{total}] 증강 ep{ep_index} "
                    f"copy{copy_idx + 1} → {n_aug} frames"
                )

    out.finalize()

    merged = LeRobotDataset(OUTPUT_REPO_ID, root=OUTPUT_ROOT)
    verify_output(source, merged, n_source, args.copies, args.skip_augmented)
    print(f"저장 완료: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
