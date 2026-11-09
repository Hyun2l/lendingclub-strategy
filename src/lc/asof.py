"""그 시점에 알 수 있던 정보만으로 학습하는 틀(미래 정보 차단).

시점 C(예: 2016-01-01)에 투자 결정을 한다면,
- 학습에는 C 이전에 발행되고, C 이전에 결과가 확정된 대출만 쓸 수 있다.
- 정상 상환은 마지막 납부 달에 확정된다.
- 부도는 마지막 납부 뒤 약 5개월(연체 150일) 후에 상각되므로 그만큼 늦게 확정된다고 본다.
"""
import numpy as np
import pandas as pd

CHARGE_OFF_LAG = 5


def months_to(d, cutoff):
    c = pd.Timestamp(cutoff)
    return ((c.year - d['ym'].dt.year) * 12 + c.month - d['ym'].dt.month).to_numpy()


def known_at(d, cutoff, lag=CHARGE_OFF_LAG):
    """cutoff 시점에 결과가 확정되어 있던 대출."""
    age = months_to(d, cutoff)
    settle = d['tau'].to_numpy() + np.where(d['bad'].to_numpy() == 1, lag, 0)
    return (age > 0) & d['resolved'].to_numpy() & (settle <= age)


def training_set(d, cutoff, lag=CHARGE_OFF_LAG):
    """cutoff 이전 발행 + cutoff 이전 확정 대출만."""
    return d[(d['ym'] < pd.Timestamp(cutoff)).to_numpy() & known_at(d, cutoff, lag)]


def walk_forward(d, years, lag=CHARGE_OFF_LAG):
    """연도 Y마다 (Y, Y년 1월 1일 기준 학습 표본, Y년 발행 대출)을 돌려준다."""
    for y in years:
        cut = pd.Timestamp(f'{y}-01-01')
        tr = training_set(d, cut, lag)
        assert (tr['ym'] < cut).all()
        yield y, tr, d[d['issue_year'] == y]


def loss_given_default(train):
    """학습 표본의 부도 대출 평균 손실률(1 - 받은 금액/대출액)."""
    b = train[train['bad'] == 1]
    return float(1 - (b['total_pymnt'] / b['funded_amnt']).mean())
