"""34b단계(진단): 34단계 E(등급 안 계좌 개설 적은 순)가 수익·Sharpe 모두 1위로 나온 것이 진짜인가.

의심 1 - 동점: 계좌 개설 수는 정수라 상위 30% 경계값과 같은 대출이 모두 승인되어 실제 승인비율이 30%를 넘을 수 있다.
         포트폴리오 초과수익은 승인 금액 비중 × (묶음 수익률 - 국채)이므로 비중이 크면 수익이 부풀려진다.
         -> 동점을 무작위로 끊어 정확히 q만 승인하고, 실제 승인 금액 비중과 '승인 묶음만의' 초과수익(비중 곱하기 전)을 함께 본다.
의심 2 - 결측: 값이 없는 대출은 승인 대상에서 빠진다. 결측 비율을 보고한다.
판정(사전 고정): 동점을 끊은 뒤에도 승인 묶음 초과수익과 포트폴리오 Sharpe가 기대수익 순보다 높으면 '단순 규칙 우위 유지'.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

sharpe = lambda x: x.mean() / x.std(ddof=1)
ac = ev['acc_open_past_24mths'].to_numpy(float)
print('계좌 개설 결측 비율(2013~16):', round(np.isnan(ac).mean(), 4))
jit = np.random.default_rng(11).random(len(ev)) * 1e-6


def top_within_grade(score, q):
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby(['issue_year', 'grade'], observed=True).indices.items():
        s = score[ix]
        ok = ~np.isnan(s)
        if ok.sum():
            a[ix[ok]] = s[ok] >= np.quantile(s[ok], 1 - q)
    return a


def pool_excess(a):
    """승인 묶음만의 발행월 초과수익(비중 곱하기 전)."""
    out = []
    for m in months:
        ix = grp[m]
        sel = ix[a[ix]]
        out.append(returns.pooled_irr(ev.iloc[sel], fee=FEE) - TB[ix].mean() if len(sel) else np.nan)
    return np.array(out)


P = ev['funded_amnt'].to_numpy(float)
for q in (0.3, 0.5):
    rules = {'단순 규칙(동점 그대로)': top_within_grade(-ac, q), '단순 규칙(동점 무작위)': top_within_grade(-ac + jit, q),
             '기대수익 순': top(ev['s_er'].to_numpy(), q), '부도확률 순': top(-ev['p'].to_numpy(), q),
             '무작위': top(np.random.default_rng(100).random(len(ev)), q)}
    print(f'\n== 목표 승인 {int(q * 100)}%')
    for nm, a in rules.items():
        port = monthly_excess(a)[0]
        pe = pool_excess(a)
        print(f'{nm}: 실제 승인 건수 비중 {a.mean():.3f}, 금액 비중 {P[a].sum() / P.sum():.3f} | '
              f'포트폴리오 수익 {100 * port.mean():.2f}%p Sharpe {sharpe(port):.2f} | 승인 묶음 초과수익 {100 * np.nanmean(pe):.2f}%p')
print(f'총 {(time.time() - t0) / 60:.1f}분')
