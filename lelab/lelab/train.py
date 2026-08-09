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

"""Training-specific helpers: the request schema and the LeRobot CLI builder.

The actual job lifecycle (subprocess management, registry, log streaming)
lives in app/jobs.py.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# 1. 정책별 기본 사전학습 체크포인트
# SmolVLA 는 사전학습 베이스에서 파인튜닝하는 것이 전제
# 지정하지 않으면 시각·언어 가중치가 무작위 초기화된 채 학습 진행
DEFAULT_POLICY_PATHS = {
    "smolvla": "lerobot/smolvla_base",
}

_IMAGE_KEY_PREFIX = "observation.images."


def resolve_base_checkpoint(
    policy_type: str, policy_path: str | None = None, *, download: bool = True
) -> str | None:
    """출발 가중치 경로 결정.

    빈 문자열은 사전학습 없이 학습하겠다는 뜻이므로 None 반환.
    download=True 면 허브 ID 를 로컬 스냅숏 경로로 바꾼다. 윈도우에서
    르-로봇이 pretrained_path 를 Path 로 변환하면서 'lerobot/smolvla_base' 가
    'lerobot\\smolvla_base' 가 되어 허브 조회에 실패하기 때문.
    """
    ref = policy_path if policy_path is not None else DEFAULT_POLICY_PATHS.get(policy_type)
    if not ref:
        return None
    if not download or Path(ref).exists():
        return ref
    from huggingface_hub import snapshot_download

    return snapshot_download(ref)


def _image_keys(features: dict) -> list[str]:
    return sorted(k for k in features if k.startswith(_IMAGE_KEY_PREFIX))


def base_expected_image_keys(base_dir: str) -> list[str]:
    """베이스 체크포인트가 기대하는 카메라 키 목록."""
    cfg = Path(base_dir) / "config.json"
    if not cfg.is_file():
        return []
    return _image_keys(json.loads(cfg.read_text(encoding="utf-8")).get("input_features") or {})


def build_rename_map(dataset_image_keys: list[str], base_image_keys: list[str]) -> dict[str, str]:
    """데이터셋 카메라 키를 베이스 키에 순서대로 대응.

    베이스(smolvla_base)는 camera1·camera2·camera3 을 기대하지만 데이터셋
    카메라 수는 다를 수 있다. 겹치는 개수만큼만 매핑하고, 이름이 이미 같으면
    항목을 만들지 않는다.
    """
    pairs = zip(sorted(dataset_image_keys), base_image_keys, strict=False)
    return {src: dst for src, dst in pairs if src != dst}


def dataset_image_keys(repo_id: str, root: str | None = None) -> list[str]:
    """데이터셋 meta/info.json 에서 카메라 키 추출.

    LeRobotDatasetMetadata 는 토치까지 끌어와 무겁고 짧은 프로세스에서
    불안정하므로, 메타데이터 파일만 직접 읽는다.
    """
    if root:
        info = Path(root) / "meta" / "info.json"
    else:
        from huggingface_hub import hf_hub_download

        info = Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset"))
    if not info.is_file():
        return []
    return _image_keys(json.loads(info.read_text(encoding="utf-8")).get("features") or {})


def prepare_pretrained_policy(request: "TrainingRequest", *, download: bool = True) -> None:
    """사전학습 베이스 사용에 필요한 값을 request 에 채운다.

    학습 실행 직전에 한 번 호출. 이미 값이 있으면 건드리지 않는다.
    """
    base = resolve_base_checkpoint(request.policy_type, request.policy_path, download=download)
    if not base:
        return
    request.policy_path = base
    if request.rename_map is not None:
        return
    expected = base_expected_image_keys(base)
    if not expected:
        return
    request.rename_map = build_rename_map(
        dataset_image_keys(request.dataset_repo_id, request.dataset_root), expected
    )


class TrainingRequest(BaseModel):
    # Dataset configuration
    dataset_repo_id: str
    dataset_revision: str | None = None
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None

    # Policy configuration
    policy_type: str = "act"
    # 출발 가중치, None 이면 DEFAULT_POLICY_PATHS 적용, 빈 문자열이면 미적용
    policy_path: str | None = None
    # 데이터셋 카메라 키를 베이스 체크포인트 키로 바꾸는 매핑
    rename_map: dict[str, str] | None = None

    # Core training parameters
    steps: int = 10000
    batch_size: int = 8
    seed: int | None = 1000
    num_workers: int = 4

    # Logging and checkpointing
    log_freq: int = 250
    save_freq: int = 1000
    eval_freq: int = 0
    save_checkpoint: bool = True

    # Output configuration
    output_dir: str = "outputs/train"
    resume: bool = False
    job_name: str | None = None

    # Weights & Biases
    wandb_enable: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = "online"
    wandb_disable_artifact: bool = False

    # Environment / evaluation
    env_type: str | None = None
    env_task: str | None = None
    eval_n_episodes: int = 10
    eval_batch_size: int = 50
    eval_use_async_envs: bool = False

    # Policy-specific
    policy_device: str | None = "cuda"
    policy_use_amp: bool = False
    # Hub upload (set by HfCloudJobRunner; not exposed in the form)
    policy_push_to_hub: bool = False
    policy_repo_id: str | None = None

    # Optimizer
    optimizer_type: str | None = "adam"
    optimizer_lr: float | None = None
    optimizer_weight_decay: float | None = None
    optimizer_grad_clip_norm: float | None = None

    # Advanced
    use_policy_training_preset: bool = True
    config_path: str | None = None


def build_training_command(
    request: TrainingRequest, output_dir: str, python_executable: str = "python"
) -> list[str]:
    """Build the argv list to invoke `<python_executable> -m lerobot.scripts.lerobot_train`.

    `output_dir` is supplied separately from the request so the caller (the
    JobRegistry) can pin it to the per-job directory rather than relying on
    request.output_dir, which the frontend doesn't even send in the new world.

    `python_executable` defaults to "python" for the cloud runner (whose
    container has lerobot on PATH); the local runner must pass sys.executable
    so the subprocess uses the same interpreter as lelab itself — otherwise
    PATH lookup picks up a different env (uv tool venv, miniforge3 base, etc.)
    that lacks lerobot.
    """
    cmd: list[str] = [python_executable, "-m", "lerobot.scripts.lerobot_train"]

    # Dataset
    cmd.extend(["--dataset.repo_id", request.dataset_repo_id])
    if request.dataset_revision:
        cmd.extend(["--dataset.revision", request.dataset_revision])
    if request.dataset_root:
        cmd.extend(["--dataset.root", request.dataset_root])
    if request.dataset_episodes:
        cmd.extend(["--dataset.episodes"] + [str(ep) for ep in request.dataset_episodes])

    # Policy
    # 2. 출발 가중치가 있으면 --policy.path, 없으면 --policy.type
    # 두 인자를 함께 넘기면 르-로봇이 체크포인트 설정과 충돌
    policy_path = request.policy_path
    if policy_path is None:
        policy_path = DEFAULT_POLICY_PATHS.get(request.policy_type)
    if policy_path:
        # 르-로봇 파서는 --policy.path=값 형식만 인식
        cmd.append(f"--policy.path={policy_path}")
        # 3. 베이스 체크포인트의 카메라 키와 데이터셋 키가 다르면 매핑 필요
        if request.rename_map:
            cmd.append(f"--rename_map={json.dumps(request.rename_map)}")
    else:
        cmd.append(f"--policy.type={request.policy_type}")

    # Core training params
    cmd.extend(["--steps", str(request.steps)])
    cmd.extend(["--batch_size", str(request.batch_size)])
    cmd.extend(["--num_workers", str(request.num_workers)])
    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])

    # Policy device / AMP / hub
    if request.policy_device:
        cmd.append(f"--policy.device={request.policy_device}")
    cmd.append(f"--policy.use_amp={'true' if request.policy_use_amp else 'false'}")
    # LeRobot defaults push_to_hub=True and demands --policy.repo_id when so.
    # Local jobs keep it off; HF Cloud jobs flip it on via the runner.
    cmd.append(f"--policy.push_to_hub={'true' if request.policy_push_to_hub else 'false'}")
    if request.policy_push_to_hub and request.policy_repo_id:
        cmd.append(f"--policy.repo_id={request.policy_repo_id}")

    # Logging / checkpointing
    cmd.extend(["--log_freq", str(request.log_freq)])
    cmd.extend(["--save_freq", str(request.save_freq)])
    cmd.extend(["--eval_freq", str(request.eval_freq)])
    cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])

    # Output
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--resume", "true" if request.resume else "false"])
    if request.job_name:
        cmd.extend(["--job_name", request.job_name])

    # W&B
    cmd.extend(["--wandb.enable", "true" if request.wandb_enable else "false"])
    if request.wandb_enable:
        if request.wandb_project:
            cmd.extend(["--wandb.project", request.wandb_project])
        if request.wandb_entity:
            cmd.extend(["--wandb.entity", request.wandb_entity])
        if request.wandb_notes:
            cmd.extend(["--wandb.notes", request.wandb_notes])
        if request.wandb_run_id:
            cmd.extend(["--wandb.run_id", request.wandb_run_id])
        if request.wandb_mode:
            cmd.extend(["--wandb.mode", request.wandb_mode])
        cmd.extend(["--wandb.disable_artifact", "true" if request.wandb_disable_artifact else "false"])

    # Env
    if request.env_type:
        cmd.extend(["--env.type", request.env_type])
    if request.env_task:
        cmd.extend(["--env.task", request.env_task])

    # Eval
    cmd.extend(["--eval.n_episodes", str(request.eval_n_episodes)])
    cmd.extend(["--eval.batch_size", str(request.eval_batch_size)])
    cmd.extend(["--eval.use_async_envs", "true" if request.eval_use_async_envs else "false"])

    # Optimizer
    if request.optimizer_type:
        cmd.extend(["--optimizer.type", request.optimizer_type])
    if request.optimizer_lr is not None:
        cmd.extend(["--optimizer.lr", str(request.optimizer_lr)])
    if request.optimizer_weight_decay is not None:
        cmd.extend(["--optimizer.weight_decay", str(request.optimizer_weight_decay)])
    if request.optimizer_grad_clip_norm is not None:
        cmd.extend(["--optimizer.grad_clip_norm", str(request.optimizer_grad_clip_norm)])

    # Advanced
    cmd.extend(["--use_policy_training_preset", "true" if request.use_policy_training_preset else "false"])
    if request.config_path:
        cmd.extend(["--config_path", request.config_path])

    return cmd
