"""ch08 평가 관측 변환 검증."""

import numpy as np
import pytest
import torch

from eval import (
    CAMERA_KEY,
    SIDE_CAMERA_KEY,
    STATE_KEY,
    TASK_INSTRUCTION,
    make_obs_batch,
    to_chw_tensor,
)


def test_to_chw_tensor_shape():
    # 1. HWC uint8 → CHW float32
    image = np.zeros((64, 48, 3), dtype=np.uint8)
    tensor = to_chw_tensor(image)
    assert tensor.shape == (3, 64, 48)
    assert tensor.dtype == torch.float32


def test_to_chw_tensor_value_range():
    # 1. 0→0.0, 255→1.0
    low = to_chw_tensor(np.zeros((4, 4, 3), dtype=np.uint8))
    high = to_chw_tensor(np.full((4, 4, 3), 255, dtype=np.uint8))
    assert float(low.min()) == pytest.approx(0.0)
    assert float(high.max()) == pytest.approx(1.0)


def test_make_obs_batch_keys():
    # 1. 정책 입력 dict 키
    obs = {
        "image": np.zeros((128, 128, 3), dtype=np.uint8),
        "side_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(6, dtype=np.float32),
    }
    batch = make_obs_batch(obs, torch.device("cpu"))
    assert CAMERA_KEY in batch
    assert SIDE_CAMERA_KEY in batch
    assert STATE_KEY in batch
    assert batch["task"] == TASK_INSTRUCTION


def test_make_obs_batch_device_cpu():
    # 1. device=cpu 텐서 위치
    obs = {
        "image": np.ones((8, 8, 3), dtype=np.uint8),
        "side_image": np.ones((8, 8, 3), dtype=np.uint8),
        "state": np.arange(6, dtype=np.float32),
    }
    batch = make_obs_batch(obs, torch.device("cpu"))
    for key in (CAMERA_KEY, SIDE_CAMERA_KEY, STATE_KEY):
        assert batch[key].device.type == "cpu"
