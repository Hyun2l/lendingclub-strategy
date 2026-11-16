"""17단계: 피처 중요도 정리. 2016년 1월 1일 기준 시점 학습 LightGBM(시드 0~4)을 2016년 발행 36개월 종결 대출에 적용해
평균 |SHAP|(로그오즈 기여)와 방향(값이 클수록 부도 위험이 올라가나)을 계산한다."""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, asof, models
from lc.config import RESULTS
from openpyxl import load_workbook
from pathlib import Path

d = data.parse(data.load())
d = d[~d['policy_fail']]
feats = data.feature_columns(d)
X = data.model_frame(d, feats)
tr = asof.training_set(d, '2016-01-01')
ev = d[(d['issue_year'] == 2016) & d['resolved'] & (d['term_m'] == 36)]
S, G = [], []
for s in range(5):
    m = models.fit_lgbm_pd(X.loc[tr.index], d.loc[tr.index, 'bad'].to_numpy(), seed=s)
    S.append(np.abs(m.booster_.predict(X.loc[ev.index], pred_contrib=True)[:, :-1]).mean(axis=0))
    G.append(m.booster_.feature_importance('gain'))
    if s == 0:
        contrib = m.booster_.predict(X.loc[ev.index], pred_contrib=True)[:, :-1]
shap = np.mean(S, axis=0)
gain = np.mean(G, axis=0)
direction = []
for j, c in enumerate(feats):
    x = X.loc[ev.index, c]
    if pd.api.types.is_numeric_dtype(x):
        ok = x.notna().to_numpy()
        r = np.corrcoef(x.to_numpy()[ok], contrib[ok, j])[0, 1] if ok.sum() > 100 and x[ok].std() > 0 else np.nan
        direction.append('높을수록 위험↑' if r > 0.2 else '높을수록 위험↓' if r < -0.2 else '비선형·약함')
    else:
        direction.append('범주형')
T = pd.DataFrame({'피처': feats, '평균|SHAP|': shap, 'SHAP 비중': shap / shap.sum(), 'gain 비중': gain / gain.sum(), '방향': direction})
T = T.sort_values('평균|SHAP|', ascending=False).reset_index(drop=True)
dic = Path.home() / 'Downloads' / 'lending_club_컬럼사전.xlsx'
if dic.exists():
    rows = list(load_workbook(dic, read_only=True)['컬럼 사전'].iter_rows(values_only=True))
    D = {r[0]: r for r in rows[1:]}
    T['그룹'] = T['피처'].map(lambda c: (D.get(c) or [None, ''])[1])
    T['한글 설명'] = T['피처'].map(lambda c: (D.get(c) or [None, '', '', '', ''])[4])
T.loc[T['피처'] == 'r', ['그룹', '한글 설명']] = ['대출 조건', '이자율(int_rate를 숫자로 변환)']
T.loc[T['피처'] == 'term_m', ['그룹', '한글 설명']] = ['대출 조건', '만기 개월 수(term을 숫자로 변환)']
PRICE = ['r', 'grade', 'sub_grade', 'term_m', 'installment']
T['LC 가격 정보'] = T['피처'].isin(PRICE)
RESULTS.mkdir(exist_ok=True)
T.round(5).to_csv(RESULTS / '17_feature_importance.csv', index=False)
print('LC 가격 정보(금리·등급·만기·월상환액) SHAP 비중', round(T.loc[T['LC 가격 정보'], 'SHAP 비중'].sum(), 3))
print('상위 10개 비중', round(T['SHAP 비중'].head(10).sum(), 3), '| 상위 20개', round(T['SHAP 비중'].head(20).sum(), 3))
grp = T.groupby('그룹')['SHAP 비중'].sum().sort_values(ascending=False)
print(grp.round(3).to_string())
pd.set_option('display.width', 220)
print(T.head(25)[['피처', 'SHAP 비중', 'gain 비중', '방향', '그룹', '한글 설명']].round(3).to_string(index=False))
