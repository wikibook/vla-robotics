#!/usr/bin/env python3

"""ch08 5단계(1k~5k) vs ch09 aug200 5단계(4k~20k) 비교."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CH09_JSON = HERE / "eval.json"

OUT_PNG = HERE / "images" / "compare.png"
OUT_MD = HERE / "compare.md"

# 1. ch08 1k 단위 5점 ↔ ch09 4× step
CH08_STEPS = [1000, 2000, 3000, 4000, 5000]
CH09_STEPS = [4000, 8000, 12000, 16000, 20000]
STAGE_LABELS = [
    "1k / 4k",
    "2k / 8k",
    "3k / 12k",
    "4k / 16k",
    "5k / 20k",
]


# 2. ch08 기준선 — 책 8장 [표 8.6] 의 3회 반복 평가 평균이다.
#    저자가 측정해 책에 실은 값이며, 이 저장소를 실행해 나온 값이 아니다.
#    ch08/02-eval-smolvla/eval.py 로 직접 재평가하면 회차마다 다른 수가 나온다
#    (시드는 블록 배치만 고정하고 파이토치 난수는 고정하지 않는다).
CH08_BASELINE = {
    1000: {"success_rate": 0.183, "avg_max_z": 0.076, "n_grasp": 4.0},
    2000: {"success_rate": 0.250, "avg_max_z": 0.095, "n_grasp": 7.0},
    3000: {"success_rate": 0.433, "avg_max_z": 0.179, "n_grasp": 10.0},
    4000: {"success_rate": 0.400, "avg_max_z": 0.153, "n_grasp": 9.7},
    5000: {"success_rate": 0.400, "avg_max_z": 0.156, "n_grasp": 9.7},
}


def load_ch08_stage(step: int) -> dict:
    return {"train_step": step, **CH08_BASELINE[step]}


def load_ch09_stages() -> list[dict]:
    if not CH09_JSON.exists():
        raise FileNotFoundError(f"{CH09_JSON} 없음 — `uv run python eval.py` 먼저 실행")
    data = json.loads(CH09_JSON.read_text())
    return data["stages"]


def plot_compare(ch08_rows: list[dict], ch09_rows: list[dict]) -> None:
    x = np.arange(len(STAGE_LABELS))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.bar(
        x - width / 2,
        [r["success_rate"] * 100 for r in ch08_rows],
        width,
        label="ch08 (50ep)",
        color="tab:gray",
    )
    ax.bar(
        x + width / 2,
        [r["success_rate"] * 100 for r in ch09_rows],
        width,
        label="ch09 aug200",
        color="tab:blue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, rotation=15)
    ax.set_ylabel("success rate (%)")
    ax.set_title("success rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - width / 2, [r["avg_max_z"] for r in ch08_rows], width, color="tab:gray")
    ax.bar(x + width / 2, [r["avg_max_z"] for r in ch09_rows], width, color="tab:blue")
    ax.axhline(0.12, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, rotation=15)
    ax.set_ylabel("avg max block z (m)")
    ax.set_title("average max z")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(x - width / 2, [r["n_grasp"] for r in ch08_rows], width, color="tab:gray")
    ax.bar(x + width / 2, [r["n_grasp"] for r in ch09_rows], width, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, rotation=15)
    ax.set_ylabel("grasp engaged (count)")
    ax.set_title("grasp activations (out of 20)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("ch08 5-stage vs ch09 aug200 5-stage (seed=42, 20 ep)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"그래프 저장: {OUT_PNG}")


def write_md(ch08_rows: list[dict], ch09_rows: list[dict]) -> None:
    lines = [
        "# ch08 vs ch09 aug200 — 5단계 비교",
        "",
        "ch08: Hub 50ep · ch09: offline aug 200ep · 평가: seed=42, 20ep",
        "",
        "| 단계 | ch08 step | ch09 step | ch08 성공률 | ch09 성공률 | "
        "ch08 avg z | ch09 avg z | ch08 grasp | ch09 grasp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, c8, c9 in zip(STAGE_LABELS, ch08_rows, ch09_rows, strict=True):
        r8 = f"{c8['success_rate'] * 100:.1f}%"
        r9 = f"{c9['success_rate'] * 100:.1f}%"
        lines.append(
            f"| {label} | {c8['train_step']} | {c9['train_step']} | {r8} | {r9} | "
            f"{c8['avg_max_z']:.3f} | {c9['avg_max_z']:.3f} | "
            f"{c8['n_grasp']:.1f}/20 | {c9['n_grasp']}/20 |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"표 저장: {OUT_MD}")


def main() -> None:
    ch08_rows = [load_ch08_stage(s) for s in CH08_STEPS]
    ch09_rows = load_ch09_stages()

    if len(ch09_rows) != len(CH08_STEPS):
        print(f"경고: ch09 stage 수 {len(ch09_rows)} (기대 {len(CH08_STEPS)})")

    print("--- ch08 기준선 (50ep, 3회 평균, 책 표 8.6) ---")
    for r in ch08_rows:
        print(
            f"step {r['train_step']:5d}  success={r['success_rate']:.1%}  "
            f"avg_z={r['avg_max_z']:.3f}  grasp={r['n_grasp']:.1f}/20"
        )
    print("--- ch09 (aug200) ---")
    for r in ch09_rows:
        print(
            f"step {r['train_step']:5d}  success={r['success_rate']:.1%}  "
            f"avg_z={r['avg_max_z']:.3f}  grasp={r['n_grasp']}/20"
        )

    plot_compare(ch08_rows, ch09_rows)
    write_md(ch08_rows, ch09_rows)


if __name__ == "__main__":
    main()
