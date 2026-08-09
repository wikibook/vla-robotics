#!/usr/bin/env python3

"""SO-ARM 101 LeRobot 데이터셋 업로드 예제."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars

LOCAL_DATASET_ID = "local/so_arm101_block_picking_main"
DATASET_ROOT = Path("./datasets")
LOCAL_DATASET_PATH = DATASET_ROOT / LOCAL_DATASET_ID
HUB_REPO_ID = "makepluscode/so_arm101_block_picking_main"
PRIVATE_REPO = False


def list_local_files(folder_path: Path) -> list[str]:
    """저장소 기준 상대 파일 경로 목록."""
    return sorted(
        path.relative_to(folder_path).as_posix()
        for path in folder_path.rglob("*")
        if path.is_file()
    )


def read_codebase_version(dataset_path: Path) -> str:
    """meta/info.json 의 LeRobot codebase_version 값."""
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    return info["codebase_version"]


def main() -> None:
    # 1. Hub 진행 표시 비활성화
    disable_progress_bars()

    # 2. 로컬 데이터셋 폴더 확인
    if not LOCAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"로컬 데이터셋 폴더를 찾을 수 없습니다: {LOCAL_DATASET_PATH}"
        )

    api = HfApi()

    # 3. Hub 데이터셋 저장소 확인
    print(f"Hub 데이터셋 저장소를 확인합니다: {HUB_REPO_ID}")
    api.create_repo(
        repo_id=HUB_REPO_ID,
        private=PRIVATE_REPO,
        repo_type="dataset",
        exist_ok=True,
    )

    # 4. 가시성 동기화 (기존 저장소도 PRIVATE_REPO 값으로 강제 전환)
    api.update_repo_settings(
        repo_id=HUB_REPO_ID,
        repo_type="dataset",
        private=PRIVATE_REPO,
    )
    print(f"저장소 가시성 동기화: private={PRIVATE_REPO}")

    # 5. 로컬과 Hub 파일 목록 비교
    print("Hub에 이미 있는 파일 목록을 가져옵니다.")
    hub_files = set(api.list_repo_files(repo_id=HUB_REPO_ID, repo_type="dataset"))
    local_files = list_local_files(LOCAL_DATASET_PATH)

    skipped_files = [path for path in local_files if path in hub_files]
    upload_files = [path for path in local_files if path not in hub_files]

    # 6. 기존 파일 건너뜀 안내
    for path in skipped_files:
        print(f"파일이 이미 존재하니 건너뜀: {path}")

    # 7. 새 파일 업로드
    if upload_files:
        print(f"새 파일 {len(upload_files)}개를 업로드합니다.")
        for path in upload_files:
            print(f"업로드 대상 파일: {path}")
        api.upload_folder(
            repo_id=HUB_REPO_ID,
            folder_path=LOCAL_DATASET_PATH,
            repo_type="dataset",
            allow_patterns=upload_files,
            commit_message="데이터셋 파일 업로드",
        )
        print("새 파일 업로드가 완료되었습니다.")
    else:
        print("업로드할 새 파일이 없습니다.")

    # 8. codebase_version 태그 동기화 (LeRobotDataset 로드 규약)
    codebase_version = read_codebase_version(LOCAL_DATASET_PATH)
    try:
        api.create_tag(
            repo_id=HUB_REPO_ID,
            tag=codebase_version,
            repo_type="dataset",
            exist_ok=True,
        )
    except TypeError:
        api.create_tag(
            repo_id=HUB_REPO_ID,
            tag=codebase_version,
            repo_type="dataset",
        )
    print(f"codebase_version 태그 동기화: {codebase_version}")

    # 9. 업로드 결과 확인
    info = api.dataset_info(repo_id=HUB_REPO_ID)

    print("업로드 확인이 완료되었습니다.")
    print(f"데이터셋 주소: https://huggingface.co/datasets/{HUB_REPO_ID}")
    print(f"데이터셋 ID: {info.id}")
    print(f"최신 커밋: {info.sha}")
    print(f"비공개 여부: {info.private}")


if __name__ == "__main__":
    main()
