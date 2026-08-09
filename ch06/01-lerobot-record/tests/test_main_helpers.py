"""ch06 main 헬퍼 함수 검증."""

import numpy as np
import pytest
import torch

from main import (
    ACTION_KEY,
    CAMERA_KEY,
    SIDE_CAMERA_KEY,
    STATE_KEY,
    TASK_LABEL,
    format_duration,
    make_frame,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (30, "30s"),
        (90, "1m 30s"),
        (3661, "1h 1m 1s"),
    ],
)
def test_format_duration(seconds: int, expected: str):
    # 1. 초 → 사람이 읽기 쉬운 문자열
    assert format_duration(float(seconds)) == expected


def test_make_frame_shapes_and_dtypes():
    # 1. fake obs → LeRobot frame 텐서 shape·dtype
    obs = {
        "image": np.zeros((128, 128, 3), dtype=np.uint8),
        "side_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(6, dtype=np.float32),
    }
    action = np.zeros(6, dtype=np.float32)
    frame = make_frame(obs, action)

    assert frame[CAMERA_KEY].shape == (3, 128, 128)
    assert frame[CAMERA_KEY].dtype == torch.float32
    assert float(frame[CAMERA_KEY].min()) >= 0.0
    assert float(frame[CAMERA_KEY].max()) <= 1.0

    assert frame[SIDE_CAMERA_KEY].shape == (3, 256, 256)
    assert frame[SIDE_CAMERA_KEY].dtype == torch.float32

    assert frame[STATE_KEY].shape == (6,)
    assert frame[STATE_KEY].dtype == torch.float32
    assert frame[ACTION_KEY].shape == (6,)
    assert frame[ACTION_KEY].dtype == torch.float32


def test_make_frame_includes_task_label():
    # 1. task 키
    obs = {
        "image": np.ones((128, 128, 3), dtype=np.uint8) * 255,
        "side_image": np.ones((256, 256, 3), dtype=np.uint8) * 128,
        "state": np.arange(6, dtype=np.float32),
    }
    frame = make_frame(obs, np.ones(6, dtype=np.float32))
    assert frame["task"] == TASK_LABEL
