"""52단계(train만, v2.5 검토): 생존 p(50단계)로 v2.4의 후보 28개(V1, V2, V4 x 10~90%, 전부 승인)를 다시 채점한다.
선택 규칙은 v2.4 그대로(건별 Sharpe 최고, 넷째 자리 동점이면 승인비율이 큰 쪽). 누수 점검용 건수도 센다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '46_sharpe_v2_4.py').read_text()
exec(SRC[:SRC.index('# ---------- 2. 후보 28개')])

P = pd.read_parquet(RESULTS / '50_p.parquet')
assert (P['ym'].to_numpy() == ev['ym'].to_numpy()).all() and np.allclose(P['p'].to_numpy(), ev['p'].to_numpy(), atol=1e-3)
rows = []
for lab, p in [('기존 p', ev['p'].to_numpy()), ('생존 p', P['p_surv'].to_numpy())]:
    sc = {'V1 기대수익 순': (1 - p) * ev['r'].to_numpy() - p * ev['lgd'].to_numpy() - TB + jit,
          'V2 부도확률 낮은 순': -p + jit,
          'V4 위험조정 순': evaluate.risk_adjusted_score(p, ev['r'], ev['lgd'], ev['tb']) + jit}
    for nm, s in sc.items():
        for q in QS9:
            rows.append({'p': lab, '전략': nm, '승인비율': q, '건별 Sharpe': evaluate.loan_sharpe(X, by_month_top(s, q))})
    rows.append({'p': lab, '전략': '전부 승인', '승인비율': 1.0, '건별 Sharpe': evaluate.loan_sharpe(X, np.ones(n, bool))})
G = pd.DataFrame(rows)
for lab in ['기존 p', '생존 p']:
    g = G[G['p'] == lab].copy()
    g['_k'] = list(zip(g['건별 Sharpe'].round(4), g['승인비율']))
    g = g.sort_values('_k', ascending=False).drop(columns='_k')
    log(f'\n== {lab}: 상위 6개\n' + g.head(6).round(4).to_string(index=False))
    log(f'{lab}: 전략별 최고\n' + g.groupby('전략')['건별 Sharpe'].max().round(4).to_string())

# 누수 점검: 라벨이 최종 상태에 기대는 경우의 건수(2016년 1월 1일 시점 예)
cut = pd.Timestamp('2016-01-01')
sub = d[(d['ym'] < cut).to_numpy()]
age = asof.months_to(sub, cut)
E = np.minimum(age - asof.CHARGE_OFF_LAG, 36)
tau, res, bad = sub['tau'].to_numpy(), sub['resolved'].to_numpy(), sub['bad'].to_numpy()
log(f'\n누수 점검(2016-01-01): 대상 {len(sub):,}건')
log(f'  사건으로 쓴 부도(tau <= E): {(res & (bad == 1) & (tau <= E)).sum():,}건')
log(f'  최종 미종결인데 마지막 납부가 E 이전(살아 있다고 둔 행): {(~res & (tau <= E)).sum():,}건')
G.round(5).to_csv(RESULTS / '52_survival_all_candidates.csv', index=False)
(RESULTS / '52_survival_all_candidates.log').write_text('\n'.join(LOG))
