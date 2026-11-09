import numpy as np
from lc import evaluate


def test_loan_sharpe_identity():
    # 거절을 초과수익 0으로 두면 SR = sqrt(a)·m / sqrt(s² + (1-a)m²) (s는 모표준편차 기준 근사)
    rng = np.random.default_rng(0)
    e = rng.normal(0.01, 0.1, 200_000)
    a = rng.random(e.size) < 0.25
    m, s, p = e[a].mean(), e[a].std(), a.mean()
    assert abs(evaluate.loan_sharpe(e, a) - np.sqrt(p) * m / np.sqrt(s**2 + (1 - p) * m**2)) < 1e-3


def test_reject_all_is_zero():
    assert evaluate.loan_sharpe(np.array([0.1, -0.2, 0.05]), np.zeros(3, bool)) == 0.0
