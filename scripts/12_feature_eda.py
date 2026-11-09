"""12단계 EDA (평가 기간 결과를 미리 보지 않게 설계).

(a) 등급 안 관계: 2007~2012년 발행 종결 대출만 사용. 같은 세부등급 안에서 각 변수가 부도를 얼마나 가르는지
    = 세부등급별 AUC의 대출 수 가중 평균(0.5 = 정보 없음). 전체 AUC와 비교하면 '등급이 이미 설명한 부분'을 뺀 정보량을 본다.
(b) 시간 변화: 결과 없이 특성 분포만. 2012년 대비 각 연도의 분포 이동(PSI, 0.1 이상이면 의미 있는 변화).
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lc import data, features
from lc.config import RESULTS

raw = data.load()
d = data.parse(raw)
d = d[~d['policy_fail']]
E = features.engineered(d, raw.loc[d.index, 'earliest_cr_line'])
RAW = ['r', 'dti', 'annual_inc', 'revol_util', 'inq_last_6mths', 'fico_range_low', 'loan_amnt', 'open_acc', 'delinq_2yrs',
       'mort_acc', 'acc_open_past_24mths', 'bc_util', 'num_tl_op_past_12m']
V = pd.concat([d[[c for c in RAW if c in d]].astype(float), E], axis=1)

h = d['resolved'] & (d['issue_year'] <= 2012)
oof, _ = features.zip3_encoding(d.loc[h, 'zip_code'], d.loc[h, 'bad'], d.loc[h, 'zip_code'])
V.loc[h, 'zip3_rate'] = oof
rows = []
for c in V.columns:
    x, yv, sg = V.loc[h, c], d.loc[h, 'bad'], d.loc[h, 'sub_grade']
    ok = x.notna()
    if ok.sum() < 1000:
        continue
    overall = roc_auc_score(yv[ok], x[ok])
    num, den = 0.0, 0
    for g, ix in pd.DataFrame({'sg': sg[ok]}).groupby('sg').indices.items():
        yy, xx = yv[ok].iloc[ix], x[ok].iloc[ix]
        if len(ix) >= 200 and yy.nunique() == 2:
            num += (roc_auc_score(yy, xx) - 0.5) * len(ix)
            den += len(ix)
    rows.append({'변수': c, '결측률': 1 - ok.mean(), '전체 AUC-0.5': overall - 0.5, '세부등급 안 AUC-0.5': num / den if den else np.nan})
A = pd.DataFrame(rows)
A['등급 안 정보 비율'] = (A['세부등급 안 AUC-0.5'] / A['전체 AUC-0.5']).where(A['전체 AUC-0.5'].abs() > 0.01)
A = A.reindex(A['세부등급 안 AUC-0.5'].abs().sort_values(ascending=False).index)
print('(a) 2007~2012 종결 대출', int(h.sum()), '건')
print(A.round(3).to_string(index=False))


def psi(ref, cur, bins=10):
    ref, cur = ref.dropna(), cur.dropna()
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return np.nan
    q[0], q[-1] = -np.inf, np.inf
    a = np.histogram(ref, q)[0] / len(ref) + 1e-6
    b = np.histogram(cur, q)[0] / len(cur) + 1e-6
    return float(((b - a) * np.log(b / a)).sum())


P = {}
for c in [c for c in V.columns if c not in ('zip3_rate', 'noise')]:
    ref = V.loc[d['issue_year'] == 2012, c]
    P[c] = {y: psi(ref, V.loc[d['issue_year'] == y, c]) for y in range(2013, 2019)}
P = pd.DataFrame(P).T
P['최대'] = P.max(axis=1)
P = P.sort_values('최대', ascending=False)
print('(b) 2012년 대비 분포 이동(PSI)')
print(P.round(3).to_string())
M = V.groupby(d['issue_year'])[['r', 'dti', 'pti', 'fico_avg', 'annual_inc', 'inq_last_6mths', 'acc_open_past_24mths']].median()
M = M.loc[2010:2018]
print('연도별 중앙값'); print(M.round(3).to_string())
RESULTS.mkdir(exist_ok=True)
A.round(4).to_csv(RESULTS / '12_within_grade_signal.csv', index=False)
P.round(4).to_csv(RESULTS / '12_drift_psi.csv')
M.round(4).to_csv(RESULTS / '12_median_by_year.csv')
