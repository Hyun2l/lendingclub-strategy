import numpy as np
from lc import calibration


def test_shift_hits_target_mean():
    p = np.random.default_rng(0).uniform(0.02, 0.4, 5000)
    c = calibration.shift_to_mean(p, 0.15)
    assert abs(calibration.apply_shift(p, c).mean() - 0.15) < 1e-6
    q = calibration.apply_shift(p, c)
    assert (np.argsort(q) == np.argsort(p)).all()  # 순서는 그대로


def test_cumulative_default_without_censoring_is_plain_rate():
    t = np.array([5, 10, 36, 36, 20])
    e = np.array([1, 2, 2, 2, 1])
    assert abs(calibration.cumulative_default(t, e, 36) - 0.4) < 1e-12


def test_cumulative_default_handles_censoring():
    # 4명 중 1명이 3개월째 부도, 1명은 2개월째 관측 중단, 나머지 2명은 5개월째 상환
    t = np.array([3, 2, 5, 5])
    e = np.array([1, 0, 2, 2])
    # 3개월 시점 위험집합 3명 중 1명 부도 -> 1/3
    assert abs(calibration.cumulative_default(t, e, 6) - 1 / 3) < 1e-12
