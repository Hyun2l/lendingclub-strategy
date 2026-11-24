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


def test_loan_sharpe_hand_example():
    # 승인 2건(+10%, -20%), 거절 1건(0) → x = [0.1, -0.2, 0]
    # 평균 -0.03333, 표본표준편차 sqrt(0.046667/2) = 0.152753 → -0.21822
    x = evaluate.loan_sharpe(np.array([0.1, -0.2, 0.3]), np.array([1, 1, 0], bool))
    assert abs(x - (-0.21822)) < 1e-5


def test_risk_adjusted_score_hand_example():
    # p 5%, 금리 8%, LGD 0.6, 국채 1.5%: s = 0.95*0.08 - 0.05*0.6 - 0.015 = 0.031
    # sigma = sqrt(0.05*0.95)*(0.08+0.6) = 0.217945*0.68 = 0.148203 → 0.031/0.148203 = 0.20917
    assert abs(evaluate.loan_sigma(0.05, 0.08, 0.6) - 0.148203) < 1e-6
    assert abs(evaluate.risk_adjusted_score(0.05, 0.08, 0.6, 0.015) - 0.20917) < 1e-5
