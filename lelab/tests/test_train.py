# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lelab.train — request schema and CLI builder."""

from __future__ import annotations

import json

import pytest


def _arg_value(cmd: list[str], flag: str) -> str:
    """Return the value passed to `--flag`. Fails the test if absent."""
    assert flag in cmd, f"{flag} missing from {cmd}"
    return cmd[cmd.index(flag) + 1]


def _eq_value(cmd: list[str], flag: str) -> str:
    """Return the value of an `--flag=value` style argument."""
    hit = [c for c in cmd if c.startswith(f"{flag}=")]
    assert hit, f"{flag}= missing from {cmd}"
    return hit[0].removeprefix(f"{flag}=")


def test_minimal_request_yields_well_formed_argv() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert _arg_value(cmd, "--dataset.repo_id") == "lerobot/pusht"
    assert _eq_value(cmd, "--policy.type") == "act"
    assert _arg_value(cmd, "--steps") == "10000"
    assert _arg_value(cmd, "--output_dir") == "/tmp/out"


def test_smolvla_defaults_to_pretrained_base() -> None:
    """SmolVLA 는 기본으로 사전학습 베이스에서 파인튜닝."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="user/pick_red", policy_type="smolvla")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert _eq_value(cmd, "--policy.path") == "lerobot/smolvla_base"
    assert "--policy.type" not in cmd


def test_explicit_policy_path_overrides_default() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/pick_red",
        policy_type="smolvla",
        policy_path="user/my_base",
    )
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert _eq_value(cmd, "--policy.path") == "user/my_base"


def test_empty_policy_path_trains_from_scratch() -> None:
    """빈 문자열은 사전학습 베이스 없이 학습."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="user/pick_red", policy_type="smolvla", policy_path="")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert "--policy.path" not in cmd
    assert _eq_value(cmd, "--policy.type") == "smolvla"


def test_act_keeps_policy_type_without_base() -> None:
    """ACT 는 사전학습 베이스가 없으므로 --policy.type 유지."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht", policy_type="act")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert _eq_value(cmd, "--policy.type") == "act"
    assert "--policy.path" not in cmd


def test_policy_path_uses_equals_form() -> None:
    """르-로봇 파서는 --policy.path=값 형식만 인식."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="user/pick_red", policy_type="smolvla")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert "--policy.path=lerobot/smolvla_base" in cmd
    assert "--policy.path" not in cmd  # 공백 분리 형식이면 실패


def test_policy_scoped_args_use_equals_form() -> None:
    """--policy.path 사용 시 다른 --policy.* 도 등호 형식이어야 값이 분리되지 않음."""
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="user/pick_red", policy_type="smolvla")
    cmd = build_training_command(req, output_dir="/tmp/out")

    for flag in ("--policy.device", "--policy.use_amp", "--policy.push_to_hub"):
        assert flag not in cmd
        assert any(c.startswith(f"{flag}=") for c in cmd)


def test_rename_map_emitted_with_policy_path() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/pick_red",
        policy_type="smolvla",
        rename_map={"observation.images.top": "observation.images.camera1"},
    )
    cmd = build_training_command(req, output_dir="/tmp/out")
    arg = next(c for c in cmd if c.startswith("--rename_map="))
    assert json.loads(arg.removeprefix("--rename_map=")) == {
        "observation.images.top": "observation.images.camera1"
    }


def test_rename_map_skipped_without_base_checkpoint() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/pick_red",
        policy_type="act",
        rename_map={"a": "b"},
    )
    cmd = build_training_command(req, output_dir="/tmp/out")
    assert not any(c.startswith("--rename_map=") for c in cmd)


def test_build_rename_map_pairs_in_order_and_skips_identical() -> None:
    from lelab.train import build_rename_map

    got = build_rename_map(
        ["observation.images.top", "observation.images.wrist"],
        ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"],
    )
    assert got == {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }

    # 이름이 이미 같으면 매핑 항목을 만들지 않음
    same = ["observation.images.camera1", "observation.images.camera2"]
    assert build_rename_map(same, same) == {}


def test_resolve_base_checkpoint_honours_opt_out_and_local_path(tmp_path) -> None:
    from lelab.train import resolve_base_checkpoint

    # 빈 문자열은 사전학습 없이 학습
    assert resolve_base_checkpoint("smolvla", "") is None
    # 베이스가 없는 정책은 None
    assert resolve_base_checkpoint("act") is None
    # 이미 로컬 경로면 내려받지 않고 그대로 사용
    assert resolve_base_checkpoint("smolvla", str(tmp_path)) == str(tmp_path)


def test_base_expected_image_keys_reads_config(tmp_path) -> None:
    from lelab.train import base_expected_image_keys

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {},
                    "observation.images.camera2": {},
                    "observation.images.camera1": {},
                }
            }
        ),
        encoding="utf-8",
    )
    assert base_expected_image_keys(str(tmp_path)) == [
        "observation.images.camera1",
        "observation.images.camera2",
    ]
    assert base_expected_image_keys(str(tmp_path / "missing")) == []


def test_optional_dataset_fields_only_present_when_set() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, "/tmp/out")
    assert "--dataset.revision" not in cmd
    assert "--dataset.root" not in cmd
    assert "--dataset.episodes" not in cmd

    req2 = TrainingRequest(
        dataset_repo_id="lerobot/pusht",
        dataset_revision="v2",
        dataset_root="/data",
        dataset_episodes=[0, 1, 2],
    )
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--dataset.revision") == "v2"
    assert _arg_value(cmd2, "--dataset.root") == "/data"
    # `--dataset.episodes` is followed by 3 string-encoded ints.
    idx = cmd2.index("--dataset.episodes")
    assert cmd2[idx + 1 : idx + 4] == ["0", "1", "2"]


def test_wandb_block_only_serialized_when_enabled() -> None:
    from lelab.train import TrainingRequest, build_training_command

    off = build_training_command(TrainingRequest(dataset_repo_id="x", wandb_enable=False), "/tmp/out")
    assert _arg_value(off, "--wandb.enable") == "false"
    assert "--wandb.project" not in off

    on = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            wandb_enable=True,
            wandb_project="proj",
            wandb_entity="me",
            wandb_run_id="abc",
        ),
        "/tmp/out",
    )
    assert _arg_value(on, "--wandb.enable") == "true"
    assert _arg_value(on, "--wandb.project") == "proj"
    assert _arg_value(on, "--wandb.entity") == "me"
    assert _arg_value(on, "--wandb.run_id") == "abc"


def test_push_to_hub_emits_repo_id_only_when_enabled() -> None:
    from lelab.train import TrainingRequest, build_training_command

    off = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=False, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _eq_value(off, "--policy.push_to_hub") == "false"
    assert "--policy.repo_id" not in off

    on = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=True, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _eq_value(on, "--policy.push_to_hub") == "true"
    assert _eq_value(on, "--policy.repo_id") == "me/x"


def test_seed_omitted_when_none() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="x", seed=None)
    cmd = build_training_command(req, "/tmp/out")
    assert "--seed" not in cmd

    req2 = TrainingRequest(dataset_repo_id="x", seed=42)
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--seed") == "42"


def test_training_request_validates_required_field() -> None:
    from pydantic import ValidationError

    from lelab.train import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest()  # dataset_repo_id is required
