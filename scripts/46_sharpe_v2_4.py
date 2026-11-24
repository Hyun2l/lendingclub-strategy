"""46단계(선택): test 프로토콜 v2.4의 선택 규칙을 train에 한 번 적용한다.

주 지표: 교수님식 건별 Sharpe. 대출 i의 x_i = 승인이면 (대출별 IRR_i - 국채 tb_i), 거절이면 0(돈은 국채).
         2013~16년 36개월 종결 대출 전체에 걸친 mean(x) / std(x). 대출별 IRR은 수수료 1% 차감(returns.loan_irr).
후보: V1 기대수익 순, V2 부도확률 낮은 순, V4 위험조정 순(s/σ) × 승인비율 10~90%, 그리고 전부 승인(100%). 발행월 컷.
선택: 건별 Sharpe가 가장 큰 후보 하나(소수 넷째 자리 동점이면 승인비율이 큰 쪽). 규칙은 docs/test_protocol_v2_4.md에 먼저 적었다.
보고만: 부트스트랩 SE와 1위와의 짝지은 차이, 같은 비율 무작위 대비, 민감도(국채 재투자 정의, 금액 가중, 포트폴리오 Sharpe),
        안정성(2013~14년으로 고른 후보를 2015~16년에 적용).
p는 30단계와 같은 시점 학습 LightGBM(시드 3개 평균).
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])

B, BLOCK = 2000, 6
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------- 1. 시점 학습 p, LGD, 점수 ----------
EV = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    ev['p'] = np.mean([fit_lgb(X0.loc[tri, usable], ytr, s).predict_proba(X0.loc[evi, usable])[:, 1] for s in SEEDS], axis=0)
    ev['lgd'] = lgd
    EV.append(ev)
    log(f'{Y}: 학습 {len(tr):,}건, 평가 {len(ev):,}건, LGD {lgd:.3f}, {(time.time() - t0) / 60:.1f}분')
ev = pd.concat(EV, ignore_index=True)
n = len(ev)
P_amt = ev['funded_amnt'].to_numpy(float)
TB = ev['tb'].to_numpy(float)
ev['s_er'] = (1 - ev['p']) * ev['r'] - ev['p'] * ev['lgd'] - ev['tb']

# 대출별 수익률 두 가지
IRR_RAW = returns.loan_irr(ev, fee=FEE).to_numpy()
IRR = np.minimum(IRR_RAW, (1 + ev['r'].to_numpy() / 12) ** 12 - 1)  # 약정 IRR 상한(v2.4 6절)
log(f'약정 IRR 상한에 걸린 대출 {(IRR_RAW > IRR).sum():,}건, 원래 최댓값 {IRR_RAW.max():.1f}')
X = IRR - TB                                    # 주 정의
R = returns.reinvested_return(ev, TB).to_numpy()
fv = P_amt * (1 + R) ** 3 * (1 - FEE)
X_RE = (fv / P_amt) ** (1 / 3) - 1 - TB         # 민감도: 국채 재투자 만기수익률(32단계 정의)
log(f'대출 {n:,}건 | 대출별 IRR 평균 {IRR.mean():.4f}, 중앙값 {np.median(IRR):.4f}, 최소 {IRR.min():.3f} | 부도 대출 평균 {IRR[ev["bad"] == 1].mean():.3f}')

jit = np.random.default_rng(11).random(n) * 1e-6
SCORES = {'V1 기대수익 순': ev['s_er'].to_numpy() + jit,
          'V2 부도확률 낮은 순': -ev['p'].to_numpy() + jit,
          'V4 위험조정 순': evaluate.risk_adjusted_score(ev['p'], ev['r'], ev['lgd'], ev['tb']) + jit}
QS9 = [i / 10 for i in range(1, 10)]

months = np.array(sorted(ev['ym'].unique()))
grp = ev.groupby('ym').indices
MI = np.zeros(n, int)
for k, m in enumerate(months):
    MI[grp[m]] = k
NM = len(months)


def by_month_top(score, q):
    a = np.zeros(n, bool)
    for ix in grp.values():
        a[ix] = score[ix] >= np.quantile(score[ix], 1 - q)
    return a


def month_sums(x):
    """발행월별 (Σx, Σx², 건수). 건별 Sharpe를 달 묶음으로 다시 뽑을 때 쓴다."""
    return (np.bincount(MI, x, NM), np.bincount(MI, x * x, NM), np.bincount(MI, minlength=NM).astype(float))


def sharpe_from(S1, S2, N):
    m = S1 / N
    v = (S2 - S1 * S1 / N) / (N - 1)
    return m / np.sqrt(v)


def portfolio_sharpe(a):
    out = []
    for k, m in enumerate(months):
        ix = grp[m]
        sel = ix[a[ix]]
        if len(sel) == 0:
            out.append(0.0)
            continue
        w = P_amt[sel].sum() / P_amt[ix].sum()
        out.append(w * (returns.pooled_irr(ev.iloc[sel], fee=FEE) - TB[ix].mean()))
    out = np.array(out)
    return out.mean() / out.std(ddof=1)


def wsharpe(x, w):
    mu = (w * x).sum() / w.sum()
    return mu / np.sqrt((w * (x - mu) ** 2).sum() / w.sum())


# ---------- 2. 후보 28개 ----------
cands = [(nm, q) for nm in SCORES for q in QS9] + [('전부 승인', 1.0)]
A, SUMS, rows = {}, {}, []
for nm, q in cands:
    a = np.ones(n, bool) if nm == '전부 승인' else by_month_top(SCORES[nm], q)
    x = np.where(a, X, 0.0)
    A[(nm, q)], SUMS[(nm, q)] = a, month_sums(x)
    rows.append({'전략': nm, '승인비율': q, '실제 승인비율': a.mean(), '금액 비중': P_amt[a].sum() / P_amt.sum(),
                 '건별 Sharpe(주)': evaluate.loan_sharpe(X, a),
                 '건별 Sharpe(국채 재투자)': evaluate.loan_sharpe(X_RE, a),
                 '건별 Sharpe(금액 가중)': wsharpe(x, P_amt),
                 '건별 Sharpe(승인만)': X[a].mean() / X[a].std(ddof=1),
                 '포트폴리오 Sharpe': portfolio_sharpe(a),
                 '승인 대출 평균 초과 IRR': X[a].mean(), '승인 부도율': ev['bad'].to_numpy()[a].mean()})
    log(f'{nm} {q:.0%}: 건별 Sharpe {rows[-1]["건별 Sharpe(주)"]:.4f}')
G = pd.DataFrame(rows)

# 같은 비율 무작위(10회 평균)
rnd_sums = {}
for q in QS9:
    draws = [by_month_top(np.random.default_rng(500 + k).random(n), q) for k in range(10)]
    G.loc[G['승인비율'] == q, '무작위 건별 Sharpe(10회)'] = np.mean([evaluate.loan_sharpe(X, a) for a in draws])
    rnd_sums[q] = [month_sums(np.where(a, X, 0.0)) for a in draws]
G.loc[G['전략'] == '전부 승인', '무작위 건별 Sharpe(10회)'] = G.loc[G['전략'] == '전부 승인', '건별 Sharpe(주)']

# ---------- 3. 선택(규칙 고정) ----------
G['_key'] = list(zip(G['건별 Sharpe(주)'].round(4), G['승인비율']))
pick_row = G.iloc[max(range(len(G)), key=lambda i: G['_key'].iloc[i])]
PICK = (pick_row['전략'], float(pick_row['승인비율']))
log(f'\n선택: {PICK[0]} {PICK[1]:.0%} (건별 Sharpe {pick_row["건별 Sharpe(주)"]:.4f})')

# ---------- 4. 부트스트랩(보고만) ----------
st = np.random.default_rng(2026).integers(0, NM - BLOCK + 1, size=(B, int(np.ceil(NM / BLOCK))))
BIX = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :NM], NM - 1)


def boot(s):
    S1, S2, N = s
    return sharpe_from(S1[BIX].sum(1), S2[BIX].sum(1), N[BIX].sum(1))


bs = {k: boot(v) for k, v in SUMS.items()}
bp = bs[PICK]
G['SE'] = [bs[(r['전략'], r['승인비율'])].std(ddof=1) for _, r in G.iterrows()]
G['선택안 - 이 후보 95% 하한'] = [np.quantile(bp - bs[(r['전략'], r['승인비율'])], 0.025) for _, r in G.iterrows()]
G['선택안과 구분됨'] = G['선택안 - 이 후보 95% 하한'] > 0
if PICK[0] != '전부 승인':
    br = np.mean([boot(s) for s in rnd_sums[PICK[1]]], axis=0)
    lo, hi = np.quantile(bp - br, [0.025, 0.975])
    log(f'train 예행: 선택안 - 같은 비율 무작위 = {pick_row["건별 Sharpe(주)"] - pick_row["무작위 건별 Sharpe(10회)"]:.4f}, 95% 구간 ({lo:.4f}, {hi:.4f})')
ba = bs[('전부 승인', 1.0)]
lo, hi = np.quantile(bp - ba, [0.025, 0.975])
log(f'train 예행: 선택안 - 전부 승인 = {pick_row["건별 Sharpe(주)"] - G.iloc[-1]["건별 Sharpe(주)"]:.4f}, 95% 구간 ({lo:.4f}, {hi:.4f})')

# ---------- 5. 안정성(보고만): 2013~14로 고르고 2015~16에 적용 ----------
val = np.array([str(m)[:4] in ('2013', '2014') for m in months])
sv = {k: sharpe_from(*(s[i][val].sum() for i in range(3))) for k, s in SUMS.items()}
so = {k: sharpe_from(*(s[i][~val].sum() for i in range(3))) for k, s in SUMS.items()}
pk = max(sv, key=lambda k: (round(sv[k], 4), k[1]))
rank_o = sorted(so, key=lambda k: -so[k]).index(pk) + 1
log(f'안정성: 2013~14로 고르면 {pk[0]} {pk[1]:.0%} (검증 {sv[pk]:.4f}) → 2015~16 {so[pk]:.4f}, 28개 중 {rank_o}위'
    f' | 전체 선택안의 2015~16 값 {so[PICK]:.4f}')
S = pd.DataFrame([{'전략': k[0], '승인비율': k[1], '2013~14 건별 Sharpe': sv[k], '2015~16 건별 Sharpe': so[k]} for k in SUMS])

# ---------- 6. 저장 ----------
G = G.drop(columns='_key')
G.round(5).to_csv(RESULTS / '46_sharpe_grid.csv', index=False)
S.round(5).to_csv(RESULTS / '46_stability.csv', index=False)
pd.DataFrame([{'전략': PICK[0], '승인비율': PICK[1], '건별 Sharpe': pick_row['건별 Sharpe(주)'],
               '무작위 건별 Sharpe': pick_row['무작위 건별 Sharpe(10회)'], 'SE': G.loc[(G['전략'] == PICK[0]) & (G['승인비율'] == PICK[1]), 'SE'].iloc[0],
               '포트폴리오 Sharpe': pick_row['포트폴리오 Sharpe'], '금액 비중': pick_row['금액 비중']}]).round(5).to_csv(RESULTS / '46_pick.csv', index=False)
pd.set_option('display.width', 250)
piv = G.pivot(index='승인비율', columns='전략', values='건별 Sharpe(주)')
log('\n== 건별 Sharpe(주 정의) 승인비율 × 전략\n' + piv.round(4).to_string())
log('\n== 민감도: 국채 재투자 정의\n' + G.pivot(index='승인비율', columns='전략', values='건별 Sharpe(국채 재투자)').round(4).to_string())
log('\n== 민감도: 포트폴리오 Sharpe\n' + G.pivot(index='승인비율', columns='전략', values='포트폴리오 Sharpe').round(3).to_string())
log(f'\n총 {(time.time() - t0) / 60:.1f}분')
(RESULTS / '46_sharpe_v2_4.log').write_text('\n'.join(LOG))
