"""50단계(train만, 보고·후보 점검): 생존(이산시간) 모델로 부도확률을 다시 만든다. test 규칙은 그대로.

문제: 기존 p는 시점 C까지 '끝난' 대출로만 학습한다. 그때까지 끝난 대출은 일찍 부도 나거나 일찍 갚은 대출에 치우치고,
      2013년 학습 표본은 11,345건뿐이다. 그래서 p의 크기가 실제의 절반 정도로 나온다.
해결: C 이전 발행 대출을 모두 쓴다. 대출마다 관측 끝 E = min(C까지 경과 개월 - 5(상각 지연), 36).
      마지막 납부 tau <= E 이고 결과가 확정된 대출은 그 구간에서 사건(부도 또는 상환 종료), 나머지는 E에서 중도절단.
      6개월 구간 6개(0-6, ..., 30-36개월)로 나눈 대출-구간 행에 LightGBM 다중분류(0 계속, 1 부도, 2 상환 종료),
      특성은 기존 101개 + 구간 번호. 일부만 관측된 구간(사건 없음)은 관측 비율을 가중치로 준다. 시드 3개 평균.
점수: 36개월 안 부도 누적확률 CIF = sum_b S_(b-1) h_d(b), S_b = S_(b-1)(1 - h_d(b) - h_p(b)).
후보(결정 47절에 미리 적음): S1 생존 p 낮은 순, S2 기존 p와 생존 p의 그달 안 순위 평균.
채택 조건: 40%에서 (후보 - V2) 95% 하한 > 0, 그리고 4개 연도 중 3개 이상에서 차이 > 0.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '46_sharpe_v2_4.py').read_text()
exec(SRC[:SRC.index('# ---------- 2. 후보 28개')])

import os
H, W, LAG = 36, 6, int(os.environ.get('SURV_LAG', asof.CHARGE_OFF_LAG))
SUF = '' if LAG == asof.CHARGE_OFF_LAG else f'_lag{LAG}'
NB = H // W
LGB_MC = dict(models.LGB_PARAMS)


def person_period(cut):
    """시점 cut에 알 수 있던 정보로 만든 대출-구간 행."""
    sub = d[(d['ym'] < pd.Timestamp(cut)).to_numpy()]
    age = asof.months_to(sub, cut)
    E = np.minimum(age - LAG, H)
    tau = sub['tau'].to_numpy()
    res = sub['resolved'].to_numpy()
    ev_ = res & (tau <= E)
    typ = np.where(ev_, np.where(sub['bad'].to_numpy() == 1, 1, 2), 0)
    T = np.where(ev_, tau, E)                       # 사건 월 또는 중도절단 월
    keep = T >= 1
    idx, T, typ = sub.index[keep], T[keep], typ[keep]
    rows_i, rows_b, rows_y, rows_w = [], [], [], []
    for b in range(NB):
        lo, hi = b * W, (b + 1) * W
        at_risk = T > lo
        ev_in = at_risk & (T <= hi) & (typ > 0)
        surv_full = at_risk & (T >= hi) & ~ev_in
        part = at_risk & (T < hi) & (typ == 0)
        for m, y, w in [(ev_in, typ, 1.0), (surv_full, 0, 1.0), (part, 0, (T - lo) / W)]:
            rows_i.append(idx[m]); rows_b.append(np.full(m.sum(), b))
            rows_y.append(np.broadcast_to(y, T.shape)[m] if np.ndim(y) else np.zeros(m.sum(), int))
            rows_w.append(np.broadcast_to(w, T.shape)[m] if np.ndim(w) else np.ones(m.sum()))
    return (np.concatenate(rows_i), np.concatenate(rows_b), np.concatenate(rows_y).astype(int),
            np.concatenate(rows_w).astype(float), len(idx), int((typ == 1).sum()))


def fit_mc(Xt, yt, wt, loan, seed):
    u = np.isin(loan, np.unique(loan)[np.random.default_rng(seed).random(len(np.unique(loan))) < 0.15])
    m = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=seed, **LGB_MC)
    m.fit(Xt[~u], yt[~u], sample_weight=wt[~u], eval_set=[(Xt[u], yt[u])], eval_sample_weight=[wt[u]],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


PS = []
for Y in range(2013, 2017):
    cut = f'{Y}-01-01'
    li, bb, yy, ww, nloan, ndef = person_period(cut)
    tr = asof.training_set(d, cut)
    usable = [c for c in feats if X0.loc[tr.index, c].nunique(dropna=True) > 1]
    Xt = X0.loc[li, usable].reset_index(drop=True)
    Xt['band'] = bb
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    Xe = pd.concat([X0.loc[evi, usable].reset_index(drop=True).assign(band=b) for b in range(NB)], ignore_index=True)
    Hs = []
    for s in SEEDS:
        Hs.append(fit_mc(Xt, yy, ww, li, s).predict_proba(Xe).reshape(NB, len(evi), 3))
    Hz = np.mean(Hs, axis=0)
    S, cif = np.ones(len(evi)), np.zeros(len(evi))
    for b in range(NB):
        cif += S * Hz[b, :, 1]
        S *= 1 - Hz[b, :, 1] - Hz[b, :, 2]
    PS.append(cif)
    log(f'{Y}: 생존 학습 대출 {nloan:,}건(기존 {len(tr):,}건), 행 {len(li):,}, 부도 사건 {ndef:,}, {(time.time() - t0) / 60:.1f}분')
ev['p_surv'] = np.concatenate(PS)
assert len(ev) == n

# ---------- 크기와 순서 ----------
grp_bad = ev['bad'].to_numpy()


def within_month_auc(p):
    return float(np.mean([roc_auc_score(grp_bad[ix], p[ix]) for ix in grp.values() if 0 < grp_bad[ix].sum() < len(ix)]))


rk = lambda p: ev.assign(_p=p).groupby('ym')['_p'].rank(pct=True).to_numpy()
ev['p_mix'] = (rk(ev['p'].to_numpy()) + rk(ev['p_surv'].to_numpy())) / 2
cal = ev.groupby('issue_year').agg(실제_부도율=('bad', 'mean'), 기존_p=('p', 'mean'), 생존_p=('p_surv', 'mean'))
log('\n== 크기(평균 예측 vs 실제)\n' + cal.round(4).to_string())
for nm, c in [('기존 p', 'p'), ('생존 p', 'p_surv'), ('순위 평균', 'p_mix')]:
    log(f'{nm}: 그달 안 AUC {within_month_auc(ev[c].to_numpy()):.4f}')

# ---------- 건별 Sharpe ----------
st = np.random.default_rng(2026).integers(0, NM - BLOCK + 1, size=(B, int(np.ceil(NM / BLOCK))))
BIX = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :NM], NM - 1)


def bs(a):
    S1, S2, N = month_sums(np.where(a, X, 0.0))
    return sharpe_from(S1[BIX].sum(1), S2[BIX].sum(1), N[BIX].sum(1))


yr = ev['issue_year'].to_numpy()
CAND = {'V2 기존 p': ev['p'], 'S1 생존 p': ev['p_surv'], 'S2 순위 평균': ev['p_mix']}
rows, A40 = [], {}
for nm, p in CAND.items():
    sc = -p.to_numpy() + jit
    for q in QS9:
        a = by_month_top(sc, q)
        rows.append({'후보': nm, '승인비율': q, '건별 Sharpe': evaluate.loan_sharpe(X, a), '승인 부도율': grp_bad[a].mean()})
        if q == 0.4:
            A40[nm] = a
G = pd.DataFrame(rows)
log('\n== 건별 Sharpe(승인비율 x 후보)\n' + G.pivot(index='승인비율', columns='후보', values='건별 Sharpe')[list(CAND)].round(4).to_string())
base = bs(A40['V2 기존 p'])
summ = []
for nm in ['S1 생존 p', 'S2 순위 평균']:
    dd = bs(A40[nm]) - base
    lo, hi = np.quantile(dd, [0.025, 0.975])
    by_y = {Y: evaluate.loan_sharpe(X[yr == Y], A40[nm][yr == Y]) - evaluate.loan_sharpe(X[yr == Y], A40['V2 기존 p'][yr == Y]) for Y in range(2013, 2017)}
    ok = lo > 0 and sum(v > 0 for v in by_y.values()) >= 3
    summ.append({'후보': nm, '40% 차이': evaluate.loan_sharpe(X, A40[nm]) - evaluate.loan_sharpe(X, A40['V2 기존 p']),
                 '95% 하한': lo, '95% 상한': hi, **{f'{Y} 차이': v for Y, v in by_y.items()}, '채택 조건': '충족' if ok else '미충족'})
S = pd.DataFrame(summ)
log('\n== 40%에서 V2 대비(6개월 블록 2000회)\n' + S.round(4).to_string(index=False))
log(f'\n총 {(time.time() - t0) / 60:.1f}분')
G.round(5).to_csv(RESULTS / f'50_survival_grid{SUF}.csv', index=False)
S.round(5).to_csv(RESULTS / f'50_survival_summary{SUF}.csv', index=False)
ev[['id', 'ym', 'p', 'p_surv', 'p_mix']].to_parquet(RESULTS / f'50_p{SUF}.parquet') if 'id' in ev else ev[['ym', 'p', 'p_surv', 'p_mix']].to_parquet(RESULTS / f'50_p{SUF}.parquet')
(RESULTS / f'50_survival_model{SUF}.log').write_text('\n'.join(LOG))
