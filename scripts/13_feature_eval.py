"""13단계: 피처 후보를 돈(선별 가치)으로 평가한다.

시점 학습(2013~2016, 연도별 1월 1일 기준)으로 LightGBM을 기본 101개 특성 + 후보로 다시 학습한다.
판정 기준(미리 정함): (1) 무작위 노이즈 피처보다 낫다 (2) 같은 모델을 다른 시드로 돌린 차이보다 크다
(3) 4개 연도 중 3개 이상 같은 방향 (4) 승인 시점 정보만 사용.
지표: AUC, 선별 가치(승인비율 10~90%에서 기대수익 순위 상위 승인 대출의 초과 IRR - 무작위 승인), 등급 안 선별 효과(상위 50%, 브린슨).
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lc import data, asof, models, evaluate, returns, features
from lc.config import RESULTS

raw = data.load()
d = data.parse(raw)
d = d[~d['policy_fail']]
pos = pd.Series(np.arange(len(d)), index=d.index)
X0 = data.model_frame(d, data.feature_columns(d))
E = features.engineered(d, raw.loc[d.index, 'earliest_cr_line'])
y = d['bad'].to_numpy()
evm = (d['resolved'] & (d['term_m'] == 36)).to_numpy()
SETS = {'기본(시드0)': ([], 0), '기본(시드1)': ([], 1), '+노이즈': (['noise'], 0), '+상환부담(pti,lti)': (['pti', 'lti'], 0),
        '+신용이력개월': (['credit_hist_m'], 0), '+근속연수': (['emp_years'], 0), '+우편번호부도율': (['zip3'], 0),
        '+FICO평균(대조군)': (['fico_avg'], 0), '+전체후보': (['pti', 'lti', 'credit_hist_m', 'emp_years', 'zip3'], 0)}
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rows = []
for Y in range(2013, 2017):
    C = pd.Timestamp(f'{Y}-01-01')
    tr = asof.training_set(d, C)
    itr = pos[tr.index].to_numpy()
    iy = np.where((d['issue_year'] == Y).to_numpy())[0]
    zt, za = features.zip3_encoding(d['zip_code'].to_numpy()[itr], y[itr], d['zip_code'].to_numpy()[iy])
    lgd = asof.loss_given_default(tr)
    ie = iy[evm[iy]]
    ev = returns.attach_treasury(d.iloc[ie]).reset_index(drop=True)
    sel = np.isin(iy, ie)
    rnd = np.random.default_rng(Y).random(len(ev))

    def curve(score):
        out = []
        for q in QS:
            a = score >= np.quantile(score, 1 - q)
            c = evaluate.cohort_irr(ev, a, fee=0.01)
            out.append((c['irr'] - c['tb']).mean())
        return np.array(out)
    base_rnd = curve(rnd)
    for nm, (extra, seed) in SETS.items():
        Xt, Xy = X0.iloc[itr].copy(), X0.iloc[iy].copy()
        for c in extra:
            if c == 'zip3':
                Xt['zip3'], Xy['zip3'] = zt, za
            else:
                Xt[c], Xy[c] = E.iloc[itr][c].to_numpy(), E.iloc[iy][c].to_numpy()
        m = models.fit_lgbm_pd(Xt, y[itr], seed=seed)
        p = m.predict_proba(Xy)[:, 1][sel]
        score = (1 - p) * ev['r'].to_numpy() - p * lgd - ev['tb'].to_numpy()
        cv = curve(score)
        a50 = score >= np.quantile(score, 0.5)
        bench = returns.pooled_irr(ev, fee=0.01)
        w = ev['funded_amnt'].to_numpy()
        within = 0.0
        for g, ix in ev.groupby('grade').indices.items():
            sa = ix[a50[ix]]
            if len(sa) > 50:
                ws = w[sa].sum() / w[a50].sum()
                within += ws * (returns.pooled_irr(ev.iloc[sa], fee=0.01) - returns.pooled_irr(ev.iloc[ix], fee=0.01))
        rows.append({'연도': Y, '피처': nm, 'AUC': roc_auc_score(ev['bad'], p), '선별 가치': (cv - base_rnd).mean(),
                     '상위 50% 초과 IRR': cv[QS.index(0.5)], '등급 안 선별 효과': within})
        print(Y, nm, round(rows[-1]['AUC'], 4), round(rows[-1]['선별 가치'], 5), flush=True)
R = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
R.round(5).to_csv(RESULTS / '13_feature_eval.csv', index=False)
base = R[R['피처'] == '기본(시드0)'].set_index('연도')
S = []
for nm in SETS:
    t = R[R['피처'] == nm].set_index('연도')
    dv = t['선별 가치'] - base['선별 가치']
    S.append({'피처': nm, 'AUC 평균': t['AUC'].mean(), 'AUC 변화': (t['AUC'] - base['AUC']).mean(),
              '선별 가치 평균': t['선별 가치'].mean(), '선별 가치 변화': dv.mean(), '개선 연도 수': int((dv > 0).sum()),
              '등급 안 선별 변화': (t['등급 안 선별 효과'] - base['등급 안 선별 효과']).mean()})
S = pd.DataFrame(S)
S.round(5).to_csv(RESULTS / '13_feature_summary.csv', index=False)
pd.set_option('display.width', 220)
print(R.round(4).to_string(index=False))
print(S.round(5).to_string(index=False))
