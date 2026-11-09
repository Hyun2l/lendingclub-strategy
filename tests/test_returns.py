import numpy as np
import pandas as pd
import pytest
from lc import returns

P, R, T = 1000.0, 0.12, 36
I = R / 12
INST = P * I / (1 - (1 + I) ** -T)
TARGET = (1 + I) ** 12 - 1  # 월 1%의 연환산 = 12.68%


def bal(k):
    return P * (1 + I) ** k - INST * ((1 + I) ** k - 1) / I


def loan(total, tau, bad=0):
    return pd.DataFrame({'installment': [INST], 'total_pymnt': [total], 'tau': [tau], 'funded_amnt': [P],
                         'loan_amnt': [P], 'term_m': [T], 'r': [R], 'bad': [bad]})


def test_on_schedule_irr_equals_contract_rate():
    d = loan(T * INST, T)
    assert returns.pooled_irr(d) == pytest.approx(TARGET, abs=1e-6)
    assert returns.reinvested_return(d, TARGET).iloc[0] == pytest.approx(TARGET, abs=1e-6)


def test_prepayment_keeps_contract_irr():
    d = loan(12 * INST + bal(12), 12)  # 12개월째 납입금 + 잔액 일시 상환
    kf, rem, n = returns.schedule(d)
    assert (kf[0], n[0]) == (12, 12) and rem[0] == pytest.approx(bal(12))
    assert returns.pooled_irr(d) == pytest.approx(TARGET, abs=1e-6)


def test_simple_annualized_understates_full_repayment():
    d = loan(T * INST, T)
    assert returns.simple_annualized(d).iloc[0] == pytest.approx((T * INST / P) ** (12 / T) - 1)
    assert returns.simple_annualized(d).iloc[0] < 0.07  # 약정 12%인데 약 6%로 계산된다


def test_default_irr_matches_hand_calculation():
    d = loan(5 * INST + 100, 5, bad=1)  # 5회 납입 후 부도, 회수금 100
    cf = np.zeros(6)
    cf[0] = -P
    cf[1:6] = INST
    cf[5] += 100
    f = lambda m: (cf / (1 + m) ** np.arange(6)).sum()
    from scipy.optimize import brentq
    assert returns.pooled_irr(d) == pytest.approx((1 + brentq(f, -0.9, 0.2)) ** 12 - 1, abs=1e-6)
    assert returns.class_return(d).iloc[0] == 0.0


def test_fee_lowers_irr():
    d = loan(T * INST, T)
    assert returns.pooled_irr(d, fee=0.01) < TARGET
