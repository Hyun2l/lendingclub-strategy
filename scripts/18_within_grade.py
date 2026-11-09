"""18단계(탐색): 후보 A '모델의 추가 수익은 같은 등급 안의 선별에서 나온다'를 반대 질문으로 검증한다.
이미 본 2013~2016년 데이터를 쓰는 탐색 분석이며, 고정한 test 프로토콜(v1)과는 무관하다.

미리 정한 기준
Q1 묶음을 바꿔도(등급, 세부등급, 금리 20구간, 대출 목적, 거주 주, 금액 5구간) 6개 중 4개 이상에서 선별 > 배분,
   그리고 금리 20구간 기준 선별이 등급 7개 기준 선별의 절반 이상이면 유지.
Q2 같은 세부등급 안에서 승인 vs 거절 대출의 특성 차이(표준화 평균차) — 설명용.
Q3 60개월 대출에서도 선별 > 배분이면 유지(종결 편향 있음).
Q4 연도별 학습 표본 크기와 배분 비중 — 설명용.
규칙: 연도별 기대수익(국채 대비) 상위 50% 승인(프로토콜의 R2와 같음). 모델: 시점 학습 LightGBM 시드 5개 평균.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, asof, models, returns
from lc.config import RESULTS, TRAIN_PARQUET

FEE = 0.01
PRED = TRAIN_PARQUET.parent / '18_predictions.parquet'
d = data.parse(data.load())
d = d[~d['policy_fail']]

if PRED.exists():
    P = pd.read_parquet(PRED)
else:
    X = data.model_frame(d, data.feature_columns(d))
    rows = []
    for Y in range(2013, 2017):
        tr = asof.training_set(d, f'{Y}-01-01')
        idx = d.index[(d['issue_year'] == Y) & d['resolved']]
        p = np.mean([models.fit_lgbm_pd(X.loc[tr.index], d.loc[tr.index, 'bad'].to_numpy(), seed=s).predict_proba(X.loc[idx])[:, 1]
                     for s in range(5)], axis=0)
        rows.append(pd.DataFrame({'p': p, 'lgd': asof.loss_given_default(tr), 'n_train': len(tr)}, index=idx))
        print(Y, '예측 완료', len(idx), flush=True)
    P = pd.concat(rows)
    P.to_parquet(PRED)


def brinson(e, a, key):
    w = e['funded_amnt'].to_numpy()
    bench = returns.pooled_irr(e, fee=FEE)
    alloc = sel = inter = 0.0
    for _, ix in e.groupby(key, observed=True).indices.items():
        wb = w[ix].sum() / w.sum()
        sa = ix[a[ix]]
        ws = w[sa].sum() / w[a].sum() if len(sa) else 0.0
        rb = returns.pooled_irr(e.iloc[ix], fee=FEE)
        rs = returns.pooled_irr(e.iloc[sa], fee=FEE) if len(sa) >= 30 else rb
        alloc += (ws - wb) * (rb - bench)
        sel += wb * (rs - rb)
        inter += (ws - wb) * (rs - rb)
    return {'차이': returns.pooled_irr(e[a], fee=FEE) - bench, '배분': alloc, '선별': sel, '상호작용': inter}


def frame(term):
    e = d.loc[P.index.intersection(d.index[d['term_m'] == term])].join(P)
    e = returns.attach_treasury(e).reset_index(drop=True)
    e['er'] = (1 - e['p']) * e['r'] - e['p'] * e['lgd'] - e['tb']
    a = np.zeros(len(e), bool)
    for _, ix in e.groupby('issue_year').indices.items():
        a[ix] = e['er'].to_numpy()[ix] >= np.quantile(e['er'].to_numpy()[ix], 0.5)
        e.loc[ix, 'rate20'] = pd.qcut(e['r'].iloc[ix].rank(method='first'), 20, labels=False)
        e.loc[ix, 'amt5'] = pd.qcut(e['loan_amnt'].iloc[ix].rank(method='first'), 5, labels=False)
    e['state'] = e['addr_state'].astype(str)
    e['purpose_'] = e['purpose'].astype(str)
    return e, a


KEYS = [('grade', '등급 7개'), ('sub_grade', '세부등급 35개'), ('rate20', '금리 20구간'), ('purpose_', '대출 목적'), ('state', '거주 주'), ('amt5', '금액 5구간')]
out = []
e36, a36 = frame(36)
for key, lab in KEYS:
    out.append({'만기': 36, '연도': '통합', '묶음': lab, **brinson(e36, a36, key)})
for Y in range(2013, 2017):
    m = (e36['issue_year'] == Y).to_numpy()
    sub = e36[m].reset_index(drop=True)
    for key, lab in [('grade', '등급 7개'), ('rate20', '금리 20구간')]:
        out.append({'만기': 36, '연도': Y, '묶음': lab, **brinson(sub, a36[m], key)})
# Q3: 60개월(2013~2015, 종결 편향 있음)
e60, a60 = frame(60)
m60 = (e60['issue_year'] <= 2015).to_numpy()
s60 = e60[m60].reset_index(drop=True)
for key, lab in [('grade', '등급 7개'), ('rate20', '금리 20구간')]:
    out.append({'만기': 60, '연도': '2013~15 통합', '묶음': lab, **brinson(s60, a60[m60], key)})
B = pd.DataFrame(out)

# Q2: 같은 세부등급 안에서 승인 vs 거절의 표준화 평균차
feat = {'연소득(로그)': np.log1p(e36['annual_inc'].astype(float)), '부채비율(dti)': e36['dti'].astype(float),
        '리볼빙 사용률': e36['revol_util'].astype(float), '최근 6개월 신용조회': e36['inq_last_6mths'].astype(float),
        '최근 24개월 개설 계좌': e36['acc_open_past_24mths'].astype(float), 'FICO 하한': e36['fico_range_low'].astype(float),
        '대출액': e36['loan_amnt'].astype(float), '금리(같은 세부등급 안)': e36['r'].astype(float),
        '모기지 보유': (e36['home_ownership'].astype(str) == 'MORTGAGE').astype(float),
        '부채 통합 목적': (e36['purpose'].astype(str) == 'debt_consolidation').astype(float),
        '근속 연수 결측': e36['emp_length'].isna().astype(float)}
smd = []
for nm, x in feat.items():
    num = den = 0.0
    for _, ix in e36.groupby('sub_grade', observed=True).indices.items():
        xa, xr = x.iloc[ix][a36[ix]], x.iloc[ix][~a36[ix]]
        if xa.notna().sum() > 30 and xr.notna().sum() > 30:
            sd = np.sqrt((xa.var() + xr.var()) / 2)
            if sd > 0:
                num += (xa.mean() - xr.mean()) / sd * len(ix)
                den += len(ix)
    smd.append({'특성': nm, '표준화 평균차(승인-거절)': num / den if den else np.nan})
S = pd.DataFrame(smd).sort_values('표준화 평균차(승인-거절)', key=abs, ascending=False)

# Q4: 학습 표본 크기와 배분 비중
q4 = B[(B['만기'] == 36) & (B['묶음'] == '등급 7개') & (B['연도'] != '통합')].copy()
q4['배분 비중'] = q4['배분'] / (q4['배분'].abs() + q4['선별'].abs())
q4['학습 표본'] = q4['연도'].map(P.groupby(d.loc[P.index, 'issue_year'])['n_train'].first())

RESULTS.mkdir(exist_ok=True)
B.round(5).to_csv(RESULTS / '18_brinson_groupings.csv', index=False)
S.round(4).to_csv(RESULTS / '18_within_grade_smd.csv', index=False)
pd.set_option('display.width', 220)
print(B.round(4).to_string(index=False))
print(S.round(3).to_string(index=False))
print(q4[['연도', '학습 표본', '배분', '선별', '배분 비중']].round(4).to_string(index=False))

# 판정
pooled = B[(B['만기'] == 36) & (B['연도'] == '통합')].set_index('묶음')
q1a = int((pooled['선별'] > pooled['배분']).sum())
q1b = pooled.loc['금리 20구간', '선별'] / pooled.loc['등급 7개', '선별']
r60 = B[B['만기'] == 60].set_index('묶음')
print(f"Q1: 선별>배분인 묶음 {q1a}/6, 금리 20구간 선별/등급 선별 = {q1b:.2f} -> {'유지' if q1a >= 4 and q1b >= 0.5 else '기각'}")
print(f"Q3: 60개월 등급 기준 선별 {r60.loc['등급 7개', '선별']:.4f} vs 배분 {r60.loc['등급 7개', '배분']:.4f} -> "
      f"{'유지' if r60.loc['등급 7개', '선별'] > r60.loc['등급 7개', '배분'] else '기각'}")
