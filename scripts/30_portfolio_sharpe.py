"""30단계(검증): 팀원 개선 제안서의 지적 검증 - 발행월별 포트폴리오 Sharpe로 보면 어떤 전략이 이기나.

포트폴리오 = 그 달 승인한 대출 + 거절한 돈(국채). 승인 금액 비중 w, 승인 묶음 수익률 R, 국채 tb이면
그 달 초과수익 = w × (R - tb). Sharpe = 48개 발행월(2013~16) 초과수익의 평균 ÷ 표준편차.
수익률 두 가지: (1) 발행월 묶음 IRR(수수료 1%), (2) 받은 돈을 국채로 만기까지 재투자한 만기 수익률(수수료 1%).

전략(승인비율 10~90%, 같은 시점 학습 LightGBM 기본 시드 3개 평균 PD):
  기대수익 순, 부도확률 낮은 순, 금리 낮은 순, 무작위(5회 평균), 그리고 단일점 기준선 전부 승인, A·B등급만.
불확실성: 시간 순서의 이동 블록 부트스트랩(블록 6개월, 2,000회).

판정(사전 고정):
- 승인비율 9개 중 과반에서 기대수익 순의 Sharpe가 금리 낮은 순보다 높고, 30%·50%에서 그 차이의 90% 구간 하한 > 0이면
  '기대수익 우위 유지'. 반대로 금리 낮은 순이 과반에서 높고 구간 상한 < 0이면 '위험 대비로는 보수적 선별 우위'.
  그 밖은 '구분 불가'. 두 수익률 정의 각각 판정한다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])

B, BLOCK = 2000, 6
EV = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    p = np.mean([fit_lgb(X0.loc[tri, usable], ytr, s).predict_proba(X0.loc[evi, usable])[:, 1] for s in SEEDS], axis=0)
    ev['p'] = p
    ev['s_er'] = (1 - p) * ev['r'] - p * lgd - ev['tb']
    R = returns.reinvested_return(ev, ev['tb'].to_numpy()).to_numpy()
    ev['fv'] = ev['funded_amnt'] * (1 + R) ** (ev['term_m'] / 12) * (1 - FEE)
    EV.append(ev)
    print(Y, f'{(time.time() - t0) / 60:.1f}분', flush=True)
ev = pd.concat(EV, ignore_index=True)
months = sorted(ev['ym'].unique())
grp = ev.groupby('ym').indices
P_amt = ev['funded_amnt'].to_numpy(float)
FV = ev['fv'].to_numpy(float)
TB = ev['tb'].to_numpy(float)


def monthly_excess(approve):
    """발행월별 포트폴리오 초과수익 (IRR 기준, 국채 재투자 기준)."""
    a = np.asarray(approve, bool)
    out_irr, out_re = [], []
    for ym in months:
        ix = grp[ym]
        sel = ix[a[ix]]
        tb = TB[ix].mean()
        w = P_amt[sel].sum() / P_amt[ix].sum()
        if len(sel) == 0:
            out_irr.append(0.0)
            out_re.append(0.0)
            continue
        r_irr = returns.pooled_irr(ev.iloc[sel], fee=FEE)
        r_re = (FV[sel].sum() / P_amt[sel].sum()) ** (12 / 36) - 1
        out_irr.append(w * (r_irr - tb))
        out_re.append(w * (r_re - tb))
    return np.array(out_irr), np.array(out_re)


def top(score, q):
    """연도별로 점수 상위 q를 승인(시점 학습과 같은 단위)."""
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby('issue_year').indices.items():
        s = score[ix]
        a[ix] = s >= np.quantile(s, 1 - q)
    return a


sharpe = lambda x: x.mean() / x.std(ddof=1)
QS9 = QS
scores = {'기대수익 순': ev['s_er'].to_numpy(), '부도확률 낮은 순': -ev['p'].to_numpy(), '금리 낮은 순': -ev['r'].to_numpy()}
series = {}
for nm, s in scores.items():
    for q in QS9:
        series[(nm, q)] = monthly_excess(top(s, q))
for q in QS9:
    draws = [monthly_excess(top(np.random.default_rng(100 + k).random(len(ev)), q)) for k in range(5)]
    series[('무작위', q)] = draws
series[('전부 승인', 1.0)] = monthly_excess(np.ones(len(ev), bool))
series[('A·B등급만', np.nan)] = monthly_excess(ev['grade'].astype(str).isin(['A', 'B']).to_numpy())
print('계산 완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)

rows = []
for (nm, q), v in series.items():
    for k, defn in enumerate(['IRR', '국채 재투자']):
        if nm == '무작위':
            sh = np.mean([sharpe(dr[k]) for dr in v])
            mu = np.mean([dr[k].mean() for dr in v])
        else:
            sh, mu = sharpe(v[k]), v[k].mean()
        rows.append({'전략': nm, '승인비율': q, '수익률 정의': defn, '포트폴리오 Sharpe': sh, '평균 초과수익': mu})
T = pd.DataFrame(rows)
ab_share = ev.loc[ev['grade'].astype(str).isin(['A', 'B']), 'funded_amnt'].sum() / ev['funded_amnt'].sum()

# 이동 블록 부트스트랩: 같은 발행월 인덱스로 짝지어 차이를 뽑는다
n = len(months)
rng = np.random.default_rng(2026)
starts = rng.integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
idx = np.concatenate([np.arange(BLOCK)[None, :] + starts[:, [j]] for j in range(starts.shape[1])], axis=1)[:, :n]


def boot_diff(x, y):
    return np.array([sharpe(x[i]) - sharpe(y[i]) for i in idx])


ci = []
for k, defn in enumerate(['IRR', '국채 재투자']):
    for q in (0.3, 0.5):
        er, lo, pd_ = series[('기대수익 순', q)][k], series[('금리 낮은 순', q)][k], series[('부도확률 낮은 순', q)][k]
        rnd = np.mean([dr[k] for dr in series[('무작위', q)]], axis=0)
        for nm, other in [('기대수익 - 금리 낮은 순', lo), ('기대수익 - 부도확률 낮은 순', pd_), ('기대수익 - 무작위', rnd)]:
            bd = boot_diff(er, other)
            ci.append({'수익률 정의': defn, '승인비율': q, '비교': nm, 'Sharpe 차이': sharpe(er) - sharpe(other),
                       '90% 하한': np.quantile(bd, 0.05), '90% 상한': np.quantile(bd, 0.95)})
C = pd.DataFrame(ci)
T.round(4).to_csv(RESULTS / '30_portfolio_sharpe.csv', index=False)
C.round(4).to_csv(RESULTS / '30_portfolio_sharpe_ci.csv', index=False)
pd.set_option('display.width', 220)
for defn in ['IRR', '국채 재투자']:
    t = T[T['수익률 정의'] == defn]
    print(f'\n== {defn}: 포트폴리오 Sharpe (행: 승인비율)')
    print(t[t['승인비율'].isin(QS9)].pivot(index='승인비율', columns='전략', values='포트폴리오 Sharpe').round(2).to_string())
    print(f'== {defn}: 평균 초과수익(포트폴리오, %p)')
    print((t[t['승인비율'].isin(QS9)].pivot(index='승인비율', columns='전략', values='평균 초과수익') * 100).round(2).to_string())
    print(t[t['전략'].isin(['전부 승인', 'A·B등급만'])][['전략', '포트폴리오 Sharpe', '평균 초과수익']].round(4).to_string(index=False))
print(f'\nA·B등급 승인 금액 비중 {ab_share:.2f}')
print(C.round(3).to_string(index=False))
for defn in ['IRR', '국채 재투자']:
    t = T[(T['수익률 정의'] == defn) & T['승인비율'].isin(QS9)].pivot(index='승인비율', columns='전략', values='포트폴리오 Sharpe')
    wins = int((t['기대수익 순'] > t['금리 낮은 순']).sum())
    c = C[(C['수익률 정의'] == defn) & (C['비교'] == '기대수익 - 금리 낮은 순')]
    if wins >= 5 and (c['90% 하한'] > 0).all():
        v = '기대수익 우위 유지'
    elif wins <= 4 and (c['90% 상한'] < 0).all():
        v = '위험 대비로는 보수적 선별 우위'
    else:
        v = '구분 불가'
    print(f'판정({defn}): 기대수익 순이 금리 낮은 순보다 Sharpe 높은 승인비율 {wins}/9 -> {v}')
print(f'총 {(time.time() - t0) / 60:.1f}분')
