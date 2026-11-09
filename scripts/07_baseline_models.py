"""7단계: 기본 모델 비교.

연도 Y(2013~2016)마다 Y년 1월 1일까지 확정된 대출로 학습하고 Y년 발행 대출에 적용한다(시점 학습).
모델: 금리만 / 로지스틱(수업 변수 + 금리·만기) / LightGBM(전체 101개 특성).
평가: 예측력(AUC), 확률 수준(예측 평균 vs 실제), 그리고 같은 승인 규칙을 적용했을 때의 투자 성과.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from lc import data, returns, asof, models, evaluate
from lc.config import RESULTS

d = data.parse(data.load())
feats = data.feature_columns(d)
X = data.model_frame(d, feats)
y = d['bad'].to_numpy()
ev_mask = (d['resolved'] & ~d['policy_fail'] & (d['term_m'] == 36)).to_numpy()
NAMES = ['금리만', '로지스틱', 'LightGBM']
P = {k: np.full(len(d), np.nan) for k in NAMES}
lgd_map, rows = {}, []
for Y, tr, _ in asof.walk_forward(d, range(2013, 2017)):
    tr = tr[~tr['policy_fail']]
    idx = np.where((d['issue_year'] == Y).to_numpy())[0]
    rt = LogisticRegression().fit(tr[['r', 'term_m']].to_numpy(float), y[tr.index])
    P['금리만'][idx] = rt.predict_proba(d.iloc[idx][['r', 'term_m']].to_numpy(float))[:, 1]
    lg = models.fit_logit_pd(tr, y[tr.index])
    P['로지스틱'][idx] = models.predict_logit(lg, d.iloc[idx])
    gb = models.fit_lgbm_pd(X.loc[tr.index], y[tr.index])
    P['LightGBM'][idx] = gb.predict_proba(X.iloc[idx])[:, 1]
    lgd_map[Y] = asof.loss_given_default(tr)
    e = idx[ev_mask[idx]]
    for k in NAMES:
        rows.append({'연도': Y, '모델': k, '학습표본': len(tr), 'AUC': roc_auc_score(y[e], P[k][e]),
                     'Brier': brier_score_loss(y[e], P[k][e]), '예측_평균부도율': P[k][e].mean(), '실제_부도율': y[e].mean()})
    print(f'{Y} 완료', flush=True)
fit_tab = pd.DataFrame(rows)
print(fit_tab.round(4).to_string(index=False))

ev = returns.attach_treasury(d[ev_mask & d['issue_year'].between(2013, 2016).to_numpy()])
evr = ev.reset_index(drop=True)
lgd = ev['issue_year'].map(lgd_map).to_numpy()
out = {'전부 승인(기준)': evaluate.summarize(evr, np.ones(len(ev), bool))}
for k in NAMES:
    p = P[k][ev.index]
    low30 = np.zeros(len(ev), bool)
    for _, ix in ev.groupby('issue_year').indices.items():
        low30[ix] = p[ix] <= np.quantile(p[ix], 0.3)
    er = (1 - p) * ev['r'].to_numpy() - p * lgd
    out[f'{k} · 부도확률 낮은 30%'] = evaluate.summarize(evr, low30)
    out[f'{k} · 기대수익 > 국채'] = evaluate.summarize(evr, er > ev['tb'].to_numpy())
# 다음 단계에서 다시 쓰도록 평가 표본의 예측값을 저장한다(데이터 폴더, 저장소에는 올리지 않음).
pred = ev[['ym', 'issue_year', 'term_m', 'r']].copy()
for k in NAMES:
    pred[f'p_{k}'] = P[k][ev.index]
pred['lgd'] = lgd
pred.to_parquet(data.TRAIN_PARQUET.parent / '07_predictions.parquet')
res = pd.DataFrame(out).T
print(res.round(4).to_string())
RESULTS.mkdir(exist_ok=True)
fit_tab.round(4).to_csv(RESULTS / '07_model_fit.csv', index=False)
res.round(4).to_csv(RESULTS / '07_model_rules.csv')
