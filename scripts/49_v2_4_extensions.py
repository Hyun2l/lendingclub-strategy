"""49단계: v2.4 보완 후보(test 전, train만). 선택안(V2 40%)은 바꾸지 않는다.

같은 건별 Sharpe(46단계 정의: 승인이면 대출별 IRR(약정 IRR 상한) - 국채, 거절이면 0)로
  1. 모델 없는 기준선: 금리 낮은 순, 세부등급 좋은 순(같은 세부등급 안은 무작위) x 승인 10~90%
  2. V5 = mu / E[x^2] 순(이 지표의 이론상 최적 순서). mu = s, E[x^2] = (1-p)(r-tb)^2 + p(LGD+tb)^2
     p는 (a) 모델 그대로, (b) 재보정: 발행연도마다 평가 대출 평균 p를 그해 학습 표본 부도율에 맞춘 배율
  3. V5 자기일관 기준선: 예측 모멘트로 c = S1/(2 S2)를 반복해 풀고, mu/E[x^2] > c 인 대출 승인(승인비율 자동)
  4. 짝지은 차이(6개월 블록 2000회, 95%): V2 40% - 금리 낮은 순 40%(H17 후보), V5 최고 - V2 40%
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '46_sharpe_v2_4.py').read_text()
SRC = SRC.replace("    ev['lgd'] = lgd\n", "    ev['lgd'] = lgd\n    ev['trb'] = tr['bad'].mean()\n")
exec(SRC[:SRC.index('# ---------- 2. 후보 28개')])
st = np.random.default_rng(2026).integers(0, NM - BLOCK + 1, size=(B, int(np.ceil(NM / BLOCK))))
BIX = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :NM], NM - 1)
boot = lambda s: sharpe_from(s[0][BIX].sum(1), s[1][BIX].sum(1), s[2][BIX].sum(1))

r, tb, lgd = ev['r'].to_numpy(), TB, ev['lgd'].to_numpy()
p_raw = ev['p'].to_numpy()
yr = ev['issue_year'].to_numpy()
scale = {Y: ev.loc[yr == Y, 'trb'].iloc[0] / p_raw[yr == Y].mean() for Y in np.unique(yr)}
p_cal = np.clip(p_raw * np.vectorize(scale.get)(yr), 0, 0.99)
log('재보정 배율(학습 부도율 / 평가 평균 p): ' + ', '.join(f'{Y} {v:.2f}' for Y, v in scale.items()))


def moments(p):
    mu = (1 - p) * r - p * lgd - tb
    e2 = (1 - p) * (r - tb) ** 2 + p * (lgd + tb) ** 2
    return mu, e2


sg = pd.Categorical(ev['sub_grade'].astype(str)).codes.astype(float)
SC = {'V2 부도확률 낮은 순': -p_raw + jit, '금리 낮은 순': -r + jit, '세부등급 좋은 순': -sg + jit}
for nm, p in [('V5 mu/E[x2] (모델 p)', p_raw), ('V5 mu/E[x2] (재보정 p)', p_cal)]:
    mu, e2 = moments(p)
    SC[nm] = mu / e2 + jit

rows, SUM = [], {}
for nm, sc in SC.items():
    for q in QS9:
        a = by_month_top(sc, q)
        SUM[(nm, q)] = month_sums(np.where(a, X, 0.0))
        rows.append({'전략': nm, '승인비율': q, '건별 Sharpe': evaluate.loan_sharpe(X, a), '승인 부도율': ev['bad'].to_numpy()[a].mean(),
                     '승인 평균 초과 IRR': X[a].mean()})
for nm, p in [('V5 자기일관 (모델 p)', p_raw), ('V5 자기일관 (재보정 p)', p_cal)]:
    mu, e2 = moments(p)
    c = 0.0
    for _ in range(50):
        a = mu / e2 > c
        c = mu[a].sum() / (2 * e2[a].sum())
    SUM[(nm, a.mean())] = month_sums(np.where(a, X, 0.0))
    rows.append({'전략': nm, '승인비율': a.mean(), '건별 Sharpe': evaluate.loan_sharpe(X, a), '승인 부도율': ev['bad'].to_numpy()[a].mean(),
                 '승인 평균 초과 IRR': X[a].mean()})
    log(f'{nm}: 기준 c = {c:.4f}, 승인 {a.mean():.1%}, 건별 Sharpe {rows[-1]["건별 Sharpe"]:.4f}')
G = pd.DataFrame(rows)
G['SE'] = [boot(SUM[(a, b)]).std(ddof=1) for a, b in zip(G['전략'], G['승인비율'])]

ref = boot(SUM[('V2 부도확률 낮은 순', 0.4)])
def pair(k):
    d = ref - boot(SUM[k])
    return np.quantile(d, [0.025, 0.975])
G['V2 40% - 이 후보 95% 하한'], G['V2 40% - 이 후보 95% 상한'] = zip(*[pair((a, b)) for a, b in zip(G['전략'], G['승인비율'])])
G.round(5).to_csv(RESULTS / '49_v2_4_extensions.csv', index=False)
pd.set_option('display.width', 250)
log('\n== 건별 Sharpe\n' + G.pivot_table(index='승인비율', columns='전략', values='건별 Sharpe').round(4).to_string())
lo, hi = pair(('금리 낮은 순', 0.4))
v2 = G.query("전략 == 'V2 부도확률 낮은 순' and 승인비율 == 0.4")['건별 Sharpe'].iloc[0]
rl = G.query("전략 == '금리 낮은 순' and 승인비율 == 0.4")['건별 Sharpe'].iloc[0]
log(f'\nH17 후보(train): V2 40% {v2:.4f} - 금리 낮은 순 40% {rl:.4f} = {v2 - rl:.4f}, 95% ({lo:.4f}, {hi:.4f})')
best = G.loc[G['전략'].str.startswith('V5')].sort_values('건별 Sharpe').iloc[-1]
log(f'V5 최고: {best["전략"]} {best["승인비율"]:.0%} {best["건별 Sharpe"]:.4f}, V2 40% - 이것 95% ({best["V2 40% - 이 후보 95% 하한"]:.4f}, {best["V2 40% - 이 후보 95% 상한"]:.4f})')
(RESULTS / '49_v2_4_extensions.log').write_text('\n'.join(LOG))
