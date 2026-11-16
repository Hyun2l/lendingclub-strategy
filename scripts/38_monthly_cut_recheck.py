"""38단계(재검증): 발행월 단위 컷으로 핵심 숫자 다시 계산 + 기준 수익률(허들) 4단계.

외부 검증에서 연도 단위 컷의 문제가 드러났으므로, 이전 결과 중 컷 단위에 기대는 것을 모두 발행월 단위로 다시 잰다.
(a) 선별 가치(25단계 1.03%p): 발행월마다 점수 상위 q vs 같은 달 무작위 q, 9개 비율 평균, 블록 부트스트랩.
(b) 두 축: 기대수익 순, 부도확률 순, 단순 규칙(발행월·등급마다 계좌 개설 적은 순), 무작위.
    허들 = 국채, 국채+1%p, 국채+2%p, BAA. 승인 30%, 50%.
동점은 모두 아주 작은 무작위 값으로 끊는다.
판정(사전 고정): (a) 90% 구간 하한 > 0 이면 '선별 가치 유지'. (b) 허들마다 Sharpe 1위 규칙과 수익 1위 규칙을 그대로 보고(판정 없음, 지도 작성).
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])
from lc import benchmarks  # noqa: E402

sharpe = lambda x: x.mean() / x.std(ddof=1)
rng = np.random.default_rng(11)
jit = rng.random(len(ev)) * 1e-6
BAA = np.asarray(benchmarks.benchmark(ev, '회사채 BAA'), float)
acc = ev['acc_open_past_24mths'].to_numpy(float)


def cut(score, q, keys):
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby(keys, observed=True).indices.items():
        s = score[ix]
        a[ix] = s >= np.quantile(s, 1 - q)
    return a


def pool_series(a, hurdle):
    """발행월별 (승인 묶음 IRR - 허들)과 포트폴리오 초과 = 비중 × 그 값."""
    pool, port = [], []
    for m in months:
        ix = grp[m]
        sel = ix[a[ix]]
        h = hurdle[ix].mean()
        r = returns.pooled_irr(ev.iloc[sel], fee=FEE) - h if len(sel) else 0.0
        pool.append(r)
        port.append(P_amt[sel].sum() / P_amt[ix].sum() * r)
    return np.array(pool), np.array(port)


# (a) 선별 가치, 발행월 컷
er = ev['s_er'].to_numpy() + jit
f = []
for q in QS:
    ps = pool_series(cut(er, q, 'ym'), TB)[0]
    rs = np.mean([pool_series(cut(np.random.default_rng(100 + k).random(len(ev)), q, 'ym'), TB)[0] for k in range(5)], axis=0)
    f.append(ps - rs)
f = np.mean(f, axis=0)
n = len(f)
st = np.random.default_rng(2026).integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
ix = np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n]
bs = f[ix].mean(axis=1)
print(f'(a) 선별 가치(발행월 컷): {100 * f.mean():.2f}%p, 90% 구간 {100 * np.quantile(bs, 0.05):.2f}~{100 * np.quantile(bs, 0.95):.2f}, 양수 달 {(f > 0).mean():.2f}', flush=True)

# (b) 허들 4단계 × 규칙
hurdles = {'국채': TB, '국채+1%p': TB + 0.01, '국채+2%p': TB + 0.02, 'BAA': BAA}
rows = []
for q in (0.3, 0.5):
    rules = {'기대수익 순': cut(er, q, 'ym'), '부도확률 순': cut(-ev['p'].to_numpy() + jit, q, 'ym'),
             '단순 규칙(계좌 개설)': cut(-acc + jit, q, ['ym', 'grade'])}
    rnd = [cut(np.random.default_rng(100 + k).random(len(ev)), q, 'ym') for k in range(5)]
    for hn, h in hurdles.items():
        for nm, a in rules.items():
            port = pool_series(a, h)[1]
            rows.append({'승인비율': q, '허들': hn, '규칙': nm, '포트폴리오 Sharpe': sharpe(port), '평균 초과(%p)': 100 * port.mean(),
                         '허들 이하 달': int((port <= 0).sum())})
        ports = [pool_series(a, h)[1] for a in rnd]
        rows.append({'승인비율': q, '허들': hn, '규칙': '무작위(5회)', '포트폴리오 Sharpe': np.mean([sharpe(p) for p in ports]),
                     '평균 초과(%p)': 100 * np.mean([p.mean() for p in ports]), '허들 이하 달': np.mean([(p <= 0).sum() for p in ports])})
    print('승인', q, '완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)
T = pd.DataFrame(rows)
T.round(3).to_csv(RESULTS / '38_monthly_cut_recheck.csv', index=False)
pd.set_option('display.width', 200)
for q in (0.3, 0.5):
    t = T[T['승인비율'] == q]
    print(f'\n== 승인 {int(q * 100)}%: Sharpe')
    print(t.pivot(index='허들', columns='규칙', values='포트폴리오 Sharpe').loc[list(hurdles)].round(2).to_string())
    print(f'== 승인 {int(q * 100)}%: 평균 초과(%p)')
    print(t.pivot(index='허들', columns='규칙', values='평균 초과(%p)').loc[list(hurdles)].round(2).to_string())
