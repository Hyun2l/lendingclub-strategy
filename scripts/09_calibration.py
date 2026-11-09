"""9단계: 부도확률 보정 비교 (LightGBM, 시점 학습).

연도 Y마다 Y년 1월 1일 기준으로만 계산한다.
- 보정 전: 7단계와 같은 모델 그대로
- 단순 보정: 학습 표본의 교차 예측값으로 isotonic
- 만기 도달 보정: 그 시점에 이미 36개월+상각 지연을 다 지켜본 36개월 대출로만 isotonic
- 생존분석 이동: 최근 4년 발행 36개월 대출(진행 중 포함)의 누적 부도율에 평균을 맞춤
평가: Y년 발행 36개월 종결 대출의 실제 부도율과 비교, 그리고 같은 승인 규칙의 투자 성과.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from lc import data, asof, models, calibration as cal, evaluate, benchmarks
from lc.config import RESULTS, TRAIN_PARQUET

d = data.parse(data.load())
d = d[~d['policy_fail']]
X = data.model_frame(d, data.feature_columns(d))
y = d['bad'].to_numpy()
pos = pd.Series(np.arange(len(d)), index=d.index)
METHODS = ['보정 전', '단순 보정', '만기 도달 보정', '생존분석 이동']
P = {k: np.full(len(d), np.nan) for k in METHODS}
LGD = {'시점': {}, '만기 도달': {}}
fit = lambda Xa, ya: models.fit_lgbm_pd(Xa, ya)
rows = []


def ece(p, yy, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': yy, 'q': q}).groupby('q')
    return float((g.size() / len(p) * (g['p'].mean() - g['y'].mean()).abs()).sum())


for Y in range(2013, 2017):
    C = pd.Timestamp(f'{Y}-01-01')
    tr = asof.training_set(d, C)
    itr = pos[tr.index].to_numpy()
    iy = np.where((d['issue_year'] == Y).to_numpy())[0]
    m = fit(X.iloc[itr], y[itr])
    p_raw = m.predict_proba(X.iloc[iy])[:, 1]
    p_oof = cal.oof_predict(fit, X.iloc[itr], y[itr])
    age_tr = asof.months_to(tr, C)
    mature = (tr['term_m'].to_numpy() == 36) & (age_tr >= 36 + asof.CHARGE_OFF_LAG)
    iso_n = cal.isotonic(p_oof, y[itr])
    iso_m = cal.isotonic(p_oof[mature], y[itr][mature])
    # 생존분석: 최근 4년 발행 36개월 대출(진행 중 포함)의 시점 기준 누적 부도율
    rec = d[(d['ym'] < C) & (d['ym'] >= C - pd.DateOffset(years=4)) & (d['term_m'] == 36)]
    age = asof.months_to(rec, C)
    settle = rec['tau'].to_numpy() + np.where(rec['bad'].to_numpy() == 1, asof.CHARGE_OFF_LAG, 0)
    known = rec['resolved'].to_numpy() & (settle <= age)
    t_ev = np.where(known, settle, np.maximum(age, 1))
    e_ev = np.where(known, np.where(rec['bad'].to_numpy() == 1, 1, 2), 0)
    target = cal.cumulative_default(t_ev, e_ev, 36)
    irec = pos[rec.index].to_numpy()
    p_rec = m.predict_proba(X.iloc[irec])[:, 1]
    in_tr = pd.Series(p_oof, index=tr.index)
    hit = rec.index.isin(tr.index)
    p_rec[hit] = in_tr[rec.index[hit]].to_numpy()
    c = cal.shift_to_mean(p_rec, target)
    P['보정 전'][iy] = p_raw
    P['단순 보정'][iy] = iso_n.predict(p_raw)
    P['만기 도달 보정'][iy] = iso_m.predict(p_raw)
    P['생존분석 이동'][iy] = cal.apply_shift(p_raw, c)
    LGD['시점'][Y] = asof.loss_given_default(tr)
    LGD['만기 도달'][Y] = asof.loss_given_default(tr[mature])
    ev = iy[(d['resolved'].to_numpy()[iy]) & (d['term_m'].to_numpy()[iy] == 36)]
    for k in METHODS:
        pk = P[k][ev]
        rows.append({'연도': Y, '방법': k, '평균 예측 부도율': pk.mean(), '실제 부도율': y[ev].mean(), 'Brier': brier_score_loss(y[ev], pk),
                     '구간별 오차(ECE)': ece(pk, y[ev]), 'AUC': roc_auc_score(y[ev], pk)})
    print(f'{Y}: 만기 도달 표본 {mature.sum():,}건, 생존분석 추정 36개월 누적 부도율 {target:.3f}, '
          f'LGD 시점 {LGD["시점"][Y]:.3f} / 만기 도달 {LGD["만기 도달"][Y]:.3f}', flush=True)

fit_tab = pd.DataFrame(rows)
print(fit_tab.round(4).to_string(index=False))

evm = (d['resolved'] & (d['term_m'] == 36) & d['issue_year'].between(2013, 2016)).to_numpy()
ev = d[evm].copy()
fl = ev[ev['bad'] == 1]
print('참고: 평가 표본의 실제 LGD(사후)', round(float(1 - (fl['total_pymnt'] / fl['funded_amnt']).mean()), 3))
out = []
for b in ['만기 국채', '만기 국채 +2%p', '회사채 BAA']:
    e = ev.copy()
    e['tb'] = benchmarks.benchmark(e, b)
    e = e.reset_index(drop=True)
    out.append({'비교 기준': b, '방법': '전부 승인', 'LGD': '-', **evaluate.summarize(e, np.ones(len(e), bool))})
    for k in METHODS:
        for lg in ['시점', '만기 도달']:
            p = P[k][evm]
            er = (1 - p) * e['r'].to_numpy() - p * e['issue_year'].map(LGD[lg]).to_numpy()
            out.append({'비교 기준': b, '방법': k, 'LGD': lg, **evaluate.summarize(e, er > e['tb'].to_numpy())})
res = pd.DataFrame(out)
keep = ['비교 기준', '방법', 'LGD', '승인비율', '코호트_평균IRR', '코호트_초과수익', '국채이하_코호트수', '코호트_Sharpe', '승인_부도율', '하위5%_평균(CVaR)']
print(res[keep].round(4).to_string(index=False))
RESULTS.mkdir(exist_ok=True)
fit_tab.round(4).to_csv(RESULTS / '09_calibration_fit.csv', index=False)
res[keep].round(4).to_csv(RESULTS / '09_calibration_rules.csv', index=False)
pd.DataFrame({k: P[k][evm] for k in METHODS}, index=ev.index).to_parquet(TRAIN_PARQUET.parent / '09_calibrated.parquet')
