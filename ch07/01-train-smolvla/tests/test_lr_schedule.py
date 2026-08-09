"""ch07 cosine warmup LR 스케줄 검증."""

import pytest

from train import TOTAL_STEPS, WARMUP_STEPS, cosine_with_warmup


def test_warmup_returns_zero_at_step_zero():
    # 1. step 0 → 0
    assert cosine_with_warmup(0) == pytest.approx(0.0)


def test_warmup_linear_ramp():
    # 1. warmup 중간 → 0.5
    assert cosine_with_warmup(WARMUP_STEPS // 2) == pytest.approx(0.5)


def test_peak_at_warmup_boundary():
    # 1. warmup 끝 → 1.0
    assert cosine_with_warmup(WARMUP_STEPS) == pytest.approx(1.0)


def test_cosine_decay_monotonic_after_warmup():
    # 1. warmup 이후 단조 감소
    values = [cosine_with_warmup(s) for s in range(WARMUP_STEPS, TOTAL_STEPS + 1)]
    for prev, curr in zip(values[:-1], values[1:]):
        assert curr <= prev + 1e-9


def test_cosine_decay_end_below_eps():
    # 1. 마지막 step → 0 근사
    assert cosine_with_warmup(TOTAL_STEPS) == pytest.approx(0.0, abs=1e-6)
