"""ch06 upload 헬퍼 검증."""

import json
from pathlib import Path

from upload import list_local_files, read_codebase_version


def test_list_local_files_returns_relative_posix(tmp_path: Path):
    # 1. 정렬된 POSIX 상대 경로
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "chunk.parquet").write_bytes(b"x")

    paths = list_local_files(tmp_path)
    assert paths == ["data/chunk.parquet", "meta/info.json"]


def test_read_codebase_version_parses_info_json(tmp_path: Path):
    # 1. meta/info.json 의 codebase_version 값 파싱
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    assert read_codebase_version(tmp_path) == "v3.0"
