"""ch09 증강 frame 추출 검증."""

from augment_dataset import (
    ACTION_KEY,
    SIDE_CAMERA_KEY,
    STATE_KEY,
    TOP_CAMERA_KEY,
    frame_from_item,
)


def test_frame_from_item_keys():
    # 1. add_frame 에 필요한 키만 추림
    item = {
        TOP_CAMERA_KEY: "top",
        SIDE_CAMERA_KEY: "side",
        STATE_KEY: "state",
        ACTION_KEY: "action",
        "task": "pick the block",
        "extra": "drop",
    }
    frame = frame_from_item(item)
    assert set(frame.keys()) == {
        TOP_CAMERA_KEY,
        SIDE_CAMERA_KEY,
        STATE_KEY,
        ACTION_KEY,
        "task",
    }
    assert frame["task"] == "pick the block"
