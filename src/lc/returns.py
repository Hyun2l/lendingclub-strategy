"""수익률 정의와 국채 연결.

월별 현금흐름은 installment, total_pymnt, last_pymnt_d(tau)로 복원한다.
정해진 납입금을 kf개월 받고, 남은 금액(조기상환 일시금 또는 부도 후 회수금)은 tau개월째에 받는다고 본다.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .config import TREASURY

H = 100  # 현금흐름 최대 개월 수


def treasury():
    """발행월·만기별 미 국채 수익률(연, 소수). 36개월=GS3, 60개월=GS5."""
    out = []
    for term, path in TREASURY.items():
        x = pd.read_csv(path)
        x.columns = ['ym', 'tb']
        x['ym'] = pd.to_datetime(x['ym'])
        x['tb'] = pd.to_numeric(x['tb'], errors='coerce') / 100
        x['term_m'] = term
        out.append(x)
    return pd.concat(out, ignore_index=True)


def attach_treasury(d):
    """d에 발행월·만기에 맞는 국채 수익률 tb를 붙인다."""
    m = d.merge(treasury(), on=['ym', 'term_m'], how='left', validate='many_to_one')
    m.index = d.index
    assert m['tb'].notna().all(), '국채 금리가 없는 발행월이 있다'
    return m


def schedule(d):
    """대출별 (정상 납입 개월 kf, 마지막 달 일시금 rem, 마지막 납부 개월 n)."""
    inst = d['installment'].to_numpy(float)
    tot = d['total_pymnt'].to_numpy(float)
    n = d['tau'].to_numpy(int)
    kf = np.minimum(n, np.floor(tot / np.maximum(inst, 1e-9))).astype(int)
    rem = np.maximum(tot - kf * inst, 0.0)
    return kf, rem, n


def class_return(d):
    """수업 정의: 정상 상환이면 약정 금리, 부도면 0."""
    return pd.Series(np.where(d['bad'] == 1, 0.0, d['r']), index=d.index)


def simple_annualized(d):
    """우승 논문 정의: (받은 총액 / 대출액)^(12/만기) - 1. 매달 돌아오는 원리금의 재투자를 무시한다."""
    hpr = (d['total_pymnt'] / d['loan_amnt']).to_numpy(float)
    return pd.Series(np.clip(hpr, 1e-6, None) ** (12 / d['term_m'].to_numpy()) - 1, index=d.index)


def reinvested_return(d, rate):
    """받은 돈을 연 rate로 만기까지 재투자했을 때의 연환산 수익률. rate는 숫자 또는 대출별 배열."""
    kf, rem, n = schedule(d)
    T = d['term_m'].to_numpy()
    P = d['funded_amnt'].to_numpy(float)
    inst = d['installment'].to_numpy(float)
    g = (1 + np.broadcast_to(np.asarray(rate, float), P.shape)) ** (1 / 12) - 1
    gs = np.where(g > 0, g, 1e-12)
    # 1..kf개월 납입금의 만기 시점 가치 = inst * ((1+g)^T - (1+g)^(T-kf)) / g
    fv = inst * ((1 + gs) ** T - (1 + gs) ** (T - kf)) / gs + rem * (1 + gs) ** (T - n)
    return pd.Series(np.clip(fv / P, 1e-6, None) ** (12 / T) - 1, index=d.index)


def cashflows(d, weight=None, fee=0.0):
    """포트폴리오의 개월별 현금흐름(0개월 = 투자, 음수). fee는 받은 돈에서 떼는 비율(LC 서비스 수수료 1%)."""
    w = np.ones(len(d)) if weight is None else np.asarray(weight, float)
    kf, rem, n = schedule(d)
    inst = d['installment'].to_numpy(float)
    cf = np.zeros(H)
    cf[0] = -(w * d['funded_amnt'].to_numpy(float)).sum()
    add = np.zeros(H + 2)
    add[1] += (w * inst).sum()
    np.add.at(add, np.minimum(kf + 1, H + 1), -(w * inst))
    cf[1:] = np.cumsum(add)[1:H]
    np.add.at(cf, np.minimum(np.maximum(n, kf), H - 1), w * rem)
    cf[1:] *= (1 - fee)
    return cf


def irr(cf):
    """월 현금흐름의 연환산 IRR."""
    cf = np.where(np.abs(cf) < 1e-6, 0.0, cf)  # 누적합의 반올림 잔여를 지운다
    f = lambda m: (cf / (1 + m) ** np.arange(len(cf))).sum()
    try:
        m = brentq(f, -0.2, 0.2)
    except ValueError:  # 손실이 매우 큰 단일 대출 등
        m = brentq(f, -0.95, 1.0)
    return (1 + m) ** 12 - 1


def pooled_irr(d, weight=None, fee=0.0):
    """여러 대출을 한 포트폴리오로 묶은 생애 IRR."""
    return irr(cashflows(d, weight, fee))
