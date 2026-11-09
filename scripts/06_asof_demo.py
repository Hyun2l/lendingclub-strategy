"""6단계: 시점 학습 틀 점검.

연도 Y(2013~2016)마다 Y년 1월 1일까지 알 수 있던 대출로 부도확률 모델을 학습하고 Y년 발행 대출에 적용한다.
- 그 시점 학습 표본의 부도율이 최종 부도율과 얼마나 다른지(종결 표본 편향)
- 시점 학습 모델의 AUC(LightGBM, 로지스틱)
- 틀 점검용 규칙 두 개의 코호트 성과(본 전략은 팀 결정 후 정한다)
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lc import data, returns, asof, models, evaluate
from lc.config import RESULTS

d = data.parse(data.load())
feats = data.feature_columns(d)
X = data.model_frame(d, feats)
y = d['bad'].to_numpy()
ev_mask = (d['resolved'] & ~d['policy_fail'] & (d['term_m'] == 36)).to_numpy()

rows, p_lgb, p_log, lgd_map = [], np.full(len(d), np.nan), np.full(len(d), np.nan), {}
for Y, tr, _ in asof.walk_forward(d, range(2013, 2017)):
    tr = tr[~tr['policy_fail']]
    idx = np.where((d['issue_year'] == Y).to_numpy())[0]
    m = models.fit_lgbm_pd(X.loc[tr.index], y[tr.index])
    p_lgb[idx] = m.predict_proba(X.iloc[idx])[:, 1]
    lg = models.fit_logit_pd(tr, y[tr.index])
    p_log[idx] = models.predict_logit(lg, d.iloc[idx])
    lgd_map[Y] = asof.loss_given_default(tr)
    final = data.resolved_sample(d[d['ym'] < pd.Timestamp(f'{Y}-01-01')])
    e = idx[ev_mask[idx]]
    rows.append({'연도': Y, '학습표본': len(tr), '그시점_부도율': tr['bad'].mean(), '최종_부도율': final['bad'].mean(),
                 'LGD': lgd_map[Y], 'AUC_LightGBM': roc_auc_score(y[e], p_lgb[e]), 'AUC_로지스틱': roc_auc_score(y[e], p_log[e])})
t = pd.DataFrame(rows).round(4)
print(t.to_string(index=False))

ev = returns.attach_treasury(d[ev_mask & d['issue_year'].between(2013, 2016).to_numpy()])
p = p_lgb[ev.index]
lgd = ev['issue_year'].map(lgd_map).to_numpy()
er = (1 - p) * ev['r'].to_numpy() - p * lgd  # 단순 기대수익(연, 근사)
rule = {'전부 승인': np.ones(len(ev), bool), '시점 PD 하위 30%(연도별)': np.zeros(len(ev), bool), '기대수익 > 국채': er > ev['tb'].to_numpy()}
for _, ix in ev.groupby('issue_year').indices.items():
    rule['시점 PD 하위 30%(연도별)'][ix] = p[ix] <= np.quantile(p[ix], 0.3)
s = pd.DataFrame({k: evaluate.summarize(ev.reset_index(drop=True), v) for k, v in rule.items()}).T.round(4)
print(s.to_string())
RESULTS.mkdir(exist_ok=True)
t.to_csv(RESULTS / '06_asof_models.csv', index=False)
s.to_csv(RESULTS / '06_asof_rules.csv')
