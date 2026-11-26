"""51단계(train만, 보고용): 승인비율을 고르는 규칙의 안정성과 연도별 유지. test 규칙(V2 40%)은 그대로.

1. 반으로 나눈 기간(2013-14 <-> 2015-16, 양방향)에서 규칙으로 q를 고르고 다른 반에 적용해 손실(그 반 최고 - 적용값)을 잰다.
   규칙: 최고값 / 평평한 구간 가운데(그 반 최고 - 1 SE 안에 드는 q의 중앙값, 가장 가까운 10% 단위, 동점이면 큰 쪽) / 고정 50%.
   SE는 그 반의 발행월 6개월 블록 부트스트랩 2000회.
2. 발행연도별 V2 40% - 같은 비율 무작위(10회 평균), - 금리 낮은 순 40%.
판정(결정 47절): 양방향 모두 손실이 작은 규칙이 있으면 말하는 방식에만 반영.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '46_sharpe_v2_4.py').read_text()
exec(SRC[:SRC.index('# ---------- 2. 후보 28개')])

yr = ev['issue_year'].to_numpy()
r = ev['r'].to_numpy()
V2 = {q: by_month_top(SCORES['V2 부도확률 낮은 순'], q) for q in QS9}
RND = {q: [by_month_top(np.random.default_rng(500 + k).random(n), q) for k in range(10)] for q in QS9}
RATE40 = by_month_top(-r + jit, 0.4)
HALVES = {'2013-14': np.isin(yr, [2013, 2014]), '2015-16': np.isin(yr, [2015, 2016])}


def half_curve(h):
    m = HALVES[h]
    mi = np.unique(MI[m])
    k = len(mi)
    st = np.random.default_rng(2026).integers(0, k - BLOCK + 1, size=(B, int(np.ceil(k / BLOCK))))
    bix = mi[np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :k], k - 1)]
    sh, se = {}, {}
    for q, a in V2.items():
        sh[q] = evaluate.loan_sharpe(X[m], a[m])
        S1, S2, N = month_sums(np.where(a, X, 0.0))
        se[q] = sharpe_from(S1[bix].sum(1), S2[bix].sum(1), N[bix].sum(1)).std()
    return pd.Series(sh), pd.Series(se)


def rules(sh, se):
    qb = sh.idxmax()
    ok = sh.index[sh >= sh[qb] - se[qb]].to_numpy()
    med = np.median(ok)
    grid = np.array(QS9)
    dist = np.abs(grid - med)
    plateau = grid[np.flatnonzero(np.isclose(dist, dist.min()))].max()
    return {'최고값': qb, '평평한 구간 가운데': round(float(plateau), 1), '고정 50%': 0.5}, ok


CUR = {h: half_curve(h) for h in HALVES}
rows = []
for src, dst in [('2013-14', '2015-16'), ('2015-16', '2013-14')]:
    ch, ok = rules(*CUR[src])
    sh_d = CUR[dst][0]
    log(f'{src}에서: 최고 {CUR[src][0].idxmax():.0%}, 1 SE 안 q {", ".join(f"{q:.0%}" for q in ok)}')
    for nm, q in ch.items():
        rows.append({'고른 기간': src, '적용 기간': dst, '규칙': nm, '고른 q': q, '적용 Sharpe': sh_d[q],
                     '적용 기간 최고': sh_d.max(), '손실': sh_d.max() - sh_d[q]})
R = pd.DataFrame(rows)
log('\n== 반으로 나눠 고르고 다른 반에 적용\n' + R.round(4).to_string(index=False))
full_rules, full_ok = rules(pd.Series({q: evaluate.loan_sharpe(X, a) for q, a in V2.items()}),
                            pd.Series({q: pd.read_csv(RESULTS / '46_sharpe_grid.csv').query("전략 == 'V2 부도확률 낮은 순' and 승인비율 == @q")['SE'].iloc[0] for q in QS9}))
log(f'\ntrain 전체: 1 SE 안 q {", ".join(f"{q:.0%}" for q in full_ok)}, 규칙별 선택 {full_rules}')

yrows = []
for Y in range(2013, 2017):
    m = yr == Y
    s2 = evaluate.loan_sharpe(X[m], V2[0.4][m])
    srnd = np.mean([evaluate.loan_sharpe(X[m], a[m]) for a in RND[0.4]])
    yrows.append({'발행연도': Y, '전체 부도율': ev['bad'].to_numpy()[m].mean(), 'V2 40%': s2, '무작위 40%': srnd,
                  '금리 낮은 순 40%': evaluate.loan_sharpe(X[m], RATE40[m]), '전부 승인': evaluate.loan_sharpe(X[m], np.ones(m.sum(), bool)),
                  'V2 - 무작위': s2 - srnd, 'V2 - 금리 낮은 순': s2 - evaluate.loan_sharpe(X[m], RATE40[m])})
Yt = pd.DataFrame(yrows)
log('\n== 발행연도별(40%)\n' + Yt.round(4).to_string(index=False))
log(f'\n총 {(time.time() - t0) / 60:.1f}분')
R.round(5).to_csv(RESULTS / '51_q_rules.csv', index=False)
Yt.round(5).to_csv(RESULTS / '51_by_year.csv', index=False)
(RESULTS / '51_q_rule_stability.log').write_text('\n'.join(LOG))
