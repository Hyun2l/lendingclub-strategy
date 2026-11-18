"""42단계: 최종 제안 변경(기대수익 순, 예산 무관) 전 train 점검.

V1 기대수익 순, V2 부도확률 낮은 순 x 승인 10~90%(발행월 컷, 국채 허들)에 대해
  - 선별 가치(승인 1달러당): 발행월 묶음 초과 IRR - 같은 비율 무작위(10회 평균), 48개월 평균
  - 포트폴리오 Sharpe와 같은 비율 무작위 대비 차이
를 6개월 블록 부트스트랩(2000회, seed 2026, 공통 인덱스) 90% 구간과 함께 낸다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

sharpe = lambda x: x.mean(axis=-1) / x.std(axis=-1, ddof=1)
rng = np.random.default_rng(11)
jit = rng.random(len(ev)) * 1e-6
er = ev['s_er'].to_numpy() + jit
lowpd = -ev['p'].to_numpy() + jit


def cut(score, q):
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby('ym', observed=True).indices.items():
        s = score[ix]
        a[ix] = s >= np.quantile(s, 1 - q)
    return a


def series(a):
    w, r = [], []
    for m in months:
        ix = grp[m]
        sel = ix[a[ix]]
        r.append(returns.pooled_irr(ev.iloc[sel], fee=FEE) - TB[ix].mean())
        w.append(P_amt[sel].sum() / P_amt[ix].sum())
    w, r = np.array(w), np.array(r)
    return w * r, r


n = len(months)
st = np.random.default_rng(2026).integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
BIX = np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n]
q90 = lambda x: (np.quantile(x, 0.05), np.quantile(x, 0.95))

rows = []
for q in QS:
    rnd = [series(cut(np.random.default_rng(500 + k).random(len(ev)), q)) for k in range(10)]
    rp = np.array([x[0] for x in rnd]); rr = np.array([x[1] for x in rnd]).mean(axis=0)
    rs, rs_bs = np.mean([sharpe(x) for x in rp]), np.mean([sharpe(x[BIX]) for x in rp], axis=0)
    for nm, sc in [('V1 기대수익 순', er), ('V2 부도확률 낮은 순', lowpd)]:
        p, r = series(cut(sc, q))
        sv = r - rr
        d_bs = sharpe(p[BIX]) - rs_bs
        rows.append({'전략': nm, '승인비율': q,
                     '선별 가치(%p)': 100 * sv.mean(), '선별 가치 하한': 100 * q90(sv[BIX].mean(axis=1))[0],
                     '선별 가치 상한': 100 * q90(sv[BIX].mean(axis=1))[1], '선별 양수 달': (sv > 0).mean(),
                     'Sharpe': sharpe(p), '무작위 Sharpe': rs, 'Sharpe 차이': sharpe(p) - rs,
                     '차이 하한': q90(d_bs)[0], '차이 상한': q90(d_bs)[1],
                     '평균 초과(%p)': 100 * p.mean(), '표준편차(%p)': 100 * p.std(ddof=1)})
    print('승인', q, f'{(time.time() - t0) / 60:.1f}분', flush=True)
C = pd.DataFrame(rows)
C.round(4).to_csv(RESULTS / '42_final_proposal_check.csv', index=False)
pd.set_option('display.width', 220)
print(C.round(2).to_string(index=False))
for nm in C['전략'].unique():
    s = C[C['전략'] == nm]
    print(f'{nm}: 10~90% 평균 선별 가치 {s["선별 가치(%p)"].mean():.2f}%p, Sharpe 차이 구간 하한>0 비율 {(s["차이 하한"] > 0).sum()}/9')
