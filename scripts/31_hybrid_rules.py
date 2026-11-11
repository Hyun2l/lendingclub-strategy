"""31단계(탐색): 수익(기대수익 순)과 위험 대비(부도확률 낮은 순)의 장점을 합친 규칙이 있나.

30단계와 같은 조건(2013~16, 48개 발행월, 시점 학습 LightGBM 시드 3개, 수익률 IRR과 국채 재투자 두 가지).
후보(모두 조율할 매개변수 없음, 결과를 보기 전에 정함):
  H1 등급 안 부도확률 낮은 순: 등급(A~G)마다 같은 비율 q만큼, 그 등급 안에서 부도확률 낮은 대출을 승인.
     등급 구성은 무작위와 같게 두고 '등급 안 선별'만 쓴다.
  H2 세부등급 안 기대수익 순: 세부등급(35개)마다 비율 q만큼, 그 안에서 기대수익 높은 대출을 승인.
  H3 위험 대비 기대수익: 점수 = 기대수익 ÷ 대출 한 건 손익의 표준편차, 표준편차 = (금리 + 손실률) × √(p(1-p)).
비교 기준: 기대수익 순, 부도확률 낮은 순, 금리 낮은 순, 무작위(30단계와 같은 정의).

판정(사전 고정): 후보가 승인비율 9개 중 5개 이상에서
  (a) 포트폴리오 Sharpe와 평균 초과수익이 모두 '부도확률 낮은 순'보다 높으면 '둘 다 개선(지배)',
  (b) Sharpe는 기대수익 순보다 높고 평균 초과수익은 무작위보다 높으면 '절충안',
  둘 다 아니면 '개선 없음'. IRR 기준으로 판정하고 국채 재투자 기준은 확인용.
이미 본 구간에서 후보 3개를 고르는 탐색이므로, 통과해도 test 프로토콜 v2의 후보로만 올린다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

sharpe = lambda x: x.mean() / x.std(ddof=1)
LGD = {}
for Y in range(2013, 2017):
    LGD[Y] = asof.loss_given_default(asof.training_set(d, f'{Y}-01-01'))
lgd_vec = ev['issue_year'].map(LGD).to_numpy()
p, r = ev['p'].to_numpy(), ev['r'].to_numpy()
er = ev['s_er'].to_numpy()
h3 = er / ((r + lgd_vec) * np.sqrt(p * (1 - p)))


def top_within(score, q, key):
    """연도 × 그룹마다 점수 상위 q를 승인."""
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby(['issue_year', key], observed=True).indices.items():
        s = score[ix]
        a[ix] = s >= np.quantile(s, 1 - q)
    return a


rules = {
    '기대수익 순': lambda q: top(er, q),
    '부도확률 낮은 순': lambda q: top(-p, q),
    '금리 낮은 순': lambda q: top(-r, q),
    'H1 등급 안 부도확률 낮은 순': lambda q: top_within(-p, q, 'grade'),
    'H2 세부등급 안 기대수익 순': lambda q: top_within(er, q, 'sub_grade'),
    'H3 위험 대비 기대수익': lambda q: top(h3, q),
}
series = {(nm, q): monthly_excess(f(q)) for nm, f in rules.items() for q in QS}
for q in QS:
    series[('무작위', q)] = [monthly_excess(top(np.random.default_rng(100 + k).random(len(ev)), q)) for k in range(5)]
print('계산 완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)

rows = []
for (nm, q), v in series.items():
    for k, defn in enumerate(['IRR', '국채 재투자']):
        if nm == '무작위':
            sh, mu = np.mean([sharpe(dr[k]) for dr in v]), np.mean([dr[k].mean() for dr in v])
        else:
            sh, mu = sharpe(v[k]), v[k].mean()
        rows.append({'전략': nm, '승인비율': q, '수익률 정의': defn, 'Sharpe': sh, '평균 초과수익': mu})
T = pd.DataFrame(rows)
T.round(4).to_csv(RESULTS / '31_hybrid_rules.csv', index=False)

n = len(months)
rng = np.random.default_rng(2026)
starts = rng.integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
idx = np.concatenate([np.arange(BLOCK)[None, :] + starts[:, [j]] for j in range(starts.shape[1])], axis=1)[:, :n]
ci = []
for nm in [k for k in rules if k.startswith('H')]:
    for q in (0.3, 0.5):
        x, y = series[(nm, q)][0], series[('부도확률 낮은 순', q)][0]
        bd = np.array([sharpe(x[i]) - sharpe(y[i]) for i in idx])
        bm = np.array([x[i].mean() - y[i].mean() for i in idx])
        ci.append({'후보': nm, '승인비율': q, 'Sharpe 차이(vs 부도확률 낮은 순)': sharpe(x) - sharpe(y),
                   'Sharpe 90% 구간': f'{np.quantile(bd, 0.05):+.2f}~{np.quantile(bd, 0.95):+.2f}',
                   '초과수익 차이(%p)': 100 * (x.mean() - y.mean()),
                   '초과수익 90% 구간': f'{100 * np.quantile(bm, 0.05):+.2f}~{100 * np.quantile(bm, 0.95):+.2f}'})
C = pd.DataFrame(ci)
C.round(3).to_csv(RESULTS / '31_hybrid_rules_ci.csv', index=False)

pd.set_option('display.width', 240)
for defn in ['IRR', '국채 재투자']:
    t = T[T['수익률 정의'] == defn]
    print(f'\n== {defn}: Sharpe')
    print(t.pivot(index='승인비율', columns='전략', values='Sharpe').round(2).to_string())
    print(f'== {defn}: 평균 초과수익(%p)')
    print((t.pivot(index='승인비율', columns='전략', values='평균 초과수익') * 100).round(2).to_string())
print()
print(C.round(3).to_string(index=False))
t = T[T['수익률 정의'] == 'IRR']
S = t.pivot(index='승인비율', columns='전략', values='Sharpe')
M = t.pivot(index='승인비율', columns='전략', values='평균 초과수익')
for nm in [k for k in rules if k.startswith('H')]:
    dom = int(((S[nm] > S['부도확률 낮은 순']) & (M[nm] > M['부도확률 낮은 순'])).sum())
    mid = int(((S[nm] > S['기대수익 순']) & (M[nm] > M['무작위'])).sum())
    v = '둘 다 개선(지배)' if dom >= 5 else ('절충안' if mid >= 5 else '개선 없음')
    print(f'판정 {nm}: 부도확률 순 지배 {dom}/9, 절충 조건 {mid}/9 -> {v}')
print(f'총 {(time.time() - t0) / 60:.1f}분')
