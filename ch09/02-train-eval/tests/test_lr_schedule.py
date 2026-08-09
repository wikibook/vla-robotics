"""ch09 cosine warmup LR 스케줄 검증."""

import pytest

from train import TOTAL_STEPS, WARMUP_STEPS, cosine_with_warmup


def test_warmup_returns_zero_at_step_zero():
    # 1. step 0 → 0
    assert cosine_with_warmup(0) == pytest.approx(0.0)


def test_peak_at_warmup_boundary():
    # 1. warmup 끝 → 1.0
    assert cosine_with_warmup(WARMUP_STEPS) == pytest.approx(1.0)


def test_cosine_decay_end_below_eps():
    # 1. 마지막 step → 0 근사
    assert cosine_with_warmup(TOTAL_STEPS) == pytest.approx(0.0, abs=1e-6)
