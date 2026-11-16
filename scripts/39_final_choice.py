"""39단계: test 규칙 v2.2의 선택 규칙(1-SE)을 train에 한 번 적용해 최종 제안을 고른다.

규칙(docs/test_protocol_v2_2.md 3절, 이 계산 전에 커밋):
  후보 = {V1 기대수익 순, V2 부도확률 낮은 순} x 승인 10~90%, 발행월 컷, 국채 허들의 발행월 포트폴리오 Sharpe.
  블록 부트스트랩(6개월, 2000회, seed 2026, 후보 공통 인덱스)으로 SE. S* - SE* 이상인 후보 중 승인비율 최대, 같으면 Sharpe 높은 쪽.
보고: 전부 승인 Sharpe, V3 단순 규칙(도전자), 무작위, 허들 4단계.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])
from lc import benchmarks  # noqa: E402

sharpe = lambda x: x.mean(axis=-1) / x.std(axis=-1, ddof=1)
rng = np.random.default_rng(11)
jit = rng.random(len(ev)) * 1e-6
BAA = np.asarray(benchmarks.benchmark(ev, '회사채 BAA'), float)
acc = ev['acc_open_past_24mths'].to_numpy(float)
er = ev['s_er'].to_numpy() + jit
lowpd = -ev['p'].to_numpy() + jit


def cut(score, q, keys):
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby(keys, observed=True).indices.items():
        s = score[ix]
        a[ix] = s >= np.quantile(s, 1 - q)
    return a


def port_series(a, hurdle):
    out = []
    for m in months:
        ix = grp[m]
        sel = ix[a[ix]]
        r = returns.pooled_irr(ev.iloc[sel], fee=FEE) - hurdle[ix].mean() if len(sel) else 0.0
        out.append(P_amt[sel].sum() / P_amt[ix].sum() * r)
    return np.array(out)


hurdles = {'국채': TB, '국채+1%p': TB + 0.01, '국채+2%p': TB + 0.02, 'BAA': BAA}
n = len(months)
st = np.random.default_rng(2026).integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
BIX = np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n]

rows, series = [], {}
for q in QS:
    for nm, sc in [('V1 기대수익 순', er), ('V2 부도확률 낮은 순', lowpd)]:
        a = cut(sc, q, 'ym')
        ps = {hn: port_series(a, h) for hn, h in hurdles.items()}
        series[(nm, q)] = ps['국채']
        rows.append({'전략': nm, '승인비율': q, 'Sharpe': sharpe(ps['국채']), 'SE': sharpe(ps['국채'][BIX]).std(ddof=1),
                     '평균 초과(%p)': 100 * ps['국채'].mean(), **{f'Sharpe({hn})': sharpe(p) for hn, p in ps.items() if hn != '국채'}})
    print('승인', q, f'{(time.time() - t0) / 60:.1f}분', flush=True)
C = pd.DataFrame(rows)

best = C.loc[C['Sharpe'].idxmax()]
ok = C[C['Sharpe'] >= best['Sharpe'] - best['SE']]
pick = ok.sort_values(['승인비율', 'Sharpe'], ascending=[False, False]).iloc[0]
C['최고와 구분 안 됨'] = C['Sharpe'] >= best['Sharpe'] - best['SE']
C['선택'] = (C['전략'] == pick['전략']) & np.isclose(C['승인비율'], pick['승인비율'])

# 보고용: 전부 승인, 같은 비율 무작위·V3, 선택안과 무작위 차이의 구간
qp = float(pick['승인비율'])
allp = port_series(np.ones(len(ev), bool), TB)
rnd = np.array([port_series(cut(np.random.default_rng(500 + k).random(len(ev)), qp, 'ym'), TB) for k in range(10)])
v3 = {q: port_series(cut(-acc + jit, q, ['ym', 'grade']), TB) for q in QS}
ch = series[(pick['전략'], qp)]
diff_bs = sharpe(ch[BIX]) - np.mean([sharpe(r[BIX]) for r in rnd], axis=0)
diff = sharpe(ch) - np.mean([sharpe(r) for r in rnd])

C.round(4).to_csv(RESULTS / '39_final_choice.csv', index=False)
pd.set_option('display.width', 200)
print('\n== 후보 18개(국채 허들, 발행월 포트폴리오 Sharpe)')
print(C.pivot(index='승인비율', columns='전략', values='Sharpe').round(2).to_string())
print('\nSE:')
print(C.pivot(index='승인비율', columns='전략', values='SE').round(2).to_string())
print(f"\n최고: {best['전략']} {int(best['승인비율'] * 100)}%  Sharpe {best['Sharpe']:.2f}, SE {best['SE']:.2f} -> 기준선 {best['Sharpe'] - best['SE']:.2f}")
print('구분 안 되는 후보:', [(r['전략'], int(round(r['승인비율'] * 100))) for _, r in ok.iterrows()])
print(f"\n최종 제안: {pick['전략']} 승인 {int(round(qp * 100))}%  Sharpe {pick['Sharpe']:.2f}, 평균 초과 {pick['평균 초과(%p)']:.2f}%p")
print('  허들별 Sharpe:', {hn: round(float(pick[f'Sharpe({hn})']), 2) for hn in ['국채+1%p', '국채+2%p', 'BAA']})
print(f'  전부 승인 Sharpe {sharpe(allp):.2f}, 같은 비율 무작위 {np.mean([sharpe(r) for r in rnd]):.2f}')
print(f'  선택안 - 무작위 Sharpe {diff:.2f}, 90% 구간 {np.quantile(diff_bs, 0.05):.2f}~{np.quantile(diff_bs, 0.95):.2f}')
print('  V3 단순 규칙 Sharpe(도전자):', {int(round(q * 100)): round(float(sharpe(s)), 2) for q, s in v3.items()})
