"""승인 결정(approve: 0/1 배열)의 성과를 여러 기준으로 잰다."""
import numpy as np
import pandas as pd

from . import returns


def loan_sharpe(excess, approve):
    """수업식 Sharpe: 승인하면 대출의 초과수익, 거절하면 국채(초과수익 0). 대출 단위 단면 평균/표준편차."""
    x = np.where(np.asarray(approve, bool), np.asarray(excess, float), 0.0)
    s = x.std(ddof=1)
    return x.mean() / s if s > 0 else 0.0


def cohort_irr(d, approve, fee=0.01):
    """발행월별로 승인 대출을 한 포트폴리오로 묶은 생애 IRR과 그 달의 국채 수익률."""
    a = np.asarray(approve, bool)
    rows = []
    for ym, idx in d.groupby('ym').indices.items():
        sel = idx[a[idx]]
        tb = d['tb'].to_numpy()[idx].mean()
        v = returns.pooled_irr(d.iloc[sel], fee=fee) if len(sel) else tb  # 전부 거절하면 국채
        rows.append((ym, v, tb, len(sel) / len(idx)))
    return pd.DataFrame(rows, columns=['ym', 'irr', 'tb', 'approve_rate']).set_index('ym')


def summarize(d, approve, fee=0.01):
    """한 전략의 요약 지표."""
    a = np.asarray(approve, bool)
    c = cohort_irr(d, a, fee)
    ex = c['irr'] - c['tb']
    sd = ex.std(ddof=1)
    return {
        '승인비율': a.mean(),
        '과제식_Sharpe(부도=0, rf5%)': loan_sharpe(returns.class_return(d) - 0.05, a),
        '대출별_Sharpe(국채재투자)': loan_sharpe(returns.reinvested_return(d, d['tb']) - d['tb'], a),
        '코호트_평균IRR': c['irr'].mean(),
        '코호트_초과수익': ex.mean(),
        '코호트_표준편차': c['irr'].std(ddof=1),
        '최저_코호트IRR': c['irr'].min(),
        '코호트_Sharpe': ex.mean() / sd if sd > 0 else np.nan,
        '국채이하_코호트수': int((ex < 0).sum()),
    }
