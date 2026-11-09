"""11단계: 지금까지의 실패에서 나온 새 비교 방법 네 가지를 같은 조건으로 검증한다.

조건: 7단계 시점 학습 예측값, 2013~2016년 발행 36개월 종결 대출, 발행월 코호트 IRR(투자자 수수료 1% 차감), 만기 국채.
미리 정한 판정 기준: (1) 기존 Sharpe와 다른 결정/통찰  (2) 무작위 승인에서 0 근처  (3) 4개 연도 중 3개 이상 같은 방향  (4) 가정이 적음

A. 포트폴리오 효용(확실성 등가): 예산 전체를 기준으로 '승인 비중 x 초과수익'의 평균과 분산. 비율 지표의 '작게 만들기' 문제를 고치는가?
B. 선별 곡선: 승인 비율 5~100%마다 초과수익을 그리고, 무작위 대비 면적으로 모델의 선별 가치를 잰다. 임계값 선택이 필요 없는가?
C. 성과 귀속(Brinson): 초과수익을 '등급 배분' 효과와 '등급 안 선별' 효과로 나눈다. 모델이 사실상 등급만 고르는가?
D. 모델 없는 규칙: 그 시점까지 만기 도달한 대출의 세부등급별 실현 초과수익 순위로 고른다. 머신러닝이 이 단순 규칙을 이기는가?
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, evaluate, benchmarks, returns
from lc.config import RESULTS, TRAIN_PARQUET

FEE = 0.01
cols = ['loan_status', 'issue_d', 'int_rate', 'term', 'grade', 'sub_grade', 'last_pymnt_d', 'installment',
        'total_pymnt', 'funded_amnt', 'loan_amnt']
full = data.parse(data.load(columns=cols))
pred = pd.read_parquet(TRAIN_PARQUET.parent / '07_predictions.parquet')
ev = full.loc[pred.index].copy()
ev['tb'] = benchmarks.benchmark(ev, '만기 국채')
ev = ev.reset_index(drop=True)
P = {k: pred[f'p_{k}'].to_numpy() for k in ['금리만', '로지스틱', 'LightGBM']}
lgd, r, tb = pred['lgd'].to_numpy(), ev['r'].to_numpy(), ev['tb'].to_numpy()
w = ev['funded_amnt'].to_numpy()
yrs = ev['issue_year'].to_numpy()
rng = np.random.default_rng(7)
SCORES = {'LightGBM 기대수익': (1 - P['LightGBM']) * r - P['LightGBM'] * lgd - tb,
          'LightGBM 부도확률(낮을수록)': -P['LightGBM'],
          '로지스틱 기대수익': (1 - P['로지스틱']) * r - P['로지스틱'] * lgd - tb,
          '금리(낮을수록)': -r,
          '무작위(플라시보)': rng.random(len(ev))}


def top_share(score, q):
    """발행연도마다 점수 상위 q 비율 승인(연도 안 분포만 사용)."""
    a = np.zeros(len(ev), bool)
    for y in np.unique(yrs):
        ix = np.where(yrs == y)[0]
        a[ix] = score[ix] >= np.quantile(score[ix], 1 - q)
    return a


def cohort_table(a):
    c = evaluate.cohort_irr(ev, a, fee=FEE)
    share = pd.Series(w * a, index=ev['ym']).groupby(level=0).sum() / pd.Series(w, index=ev['ym']).groupby(level=0).sum()
    c['alpha'] = share.reindex(c.index).to_numpy()
    c['port_excess'] = c['alpha'] * (c['irr'] - c['tb'])  # 예산 전체 기준 초과수익
    c['year'] = c.index.year
    return c


# ---------- A + B: 선별 곡선과 효용 ----------
QS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
curve = []
for nm, s in SCORES.items():
    for q in QS:
        c = cohort_table(top_share(s, q) if q < 1 else np.ones(len(ev), bool))
        x = c['port_excess']
        row = {'점수': nm, '승인비율': q, '승인 대출 초과수익': (c['irr'] - c['tb']).mean(), '예산 기준 초과수익': x.mean(),
               '예산 기준 변동(표준편차)': x.std(ddof=1), '시간 Sharpe': x.mean() / x.std(ddof=1)}
        for g in [2, 10, 50]:
            row[f'확실성등가 γ={g}'] = x.mean() - g / 2 * x.var(ddof=1)
        for y in range(2013, 2017):
            row[f'{y} 승인 초과수익'] = (c.loc[c['year'] == y, 'irr'] - c.loc[c['year'] == y, 'tb']).mean()
        curve.append(row)
    print('곡선', nm, flush=True)
C = pd.DataFrame(curve)
pd.set_option('display.width', 260)
print(C.round(4).to_string(index=False))

# 선별 가치 = 무작위 곡선 대비 '승인 대출 초과수익' 차이의 평균(5~90% 구간), 연도별
area = []
rnd = C[C['점수'] == '무작위(플라시보)'].set_index('승인비율')
for nm in SCORES:
    m = C[C['점수'] == nm].set_index('승인비율')
    qq = [q for q in QS if q < 1]
    row = {'점수': nm, '선별 가치(전체)': (m.loc[qq, '승인 대출 초과수익'] - rnd.loc[qq, '승인 대출 초과수익']).mean()}
    for y in range(2013, 2017):
        row[f'{y}'] = (m.loc[qq, f'{y} 승인 초과수익'] - rnd.loc[qq, f'{y} 승인 초과수익']).mean()
    row['양수 연도 수'] = sum(row[f'{y}'] > 0 for y in range(2013, 2017))
    area.append(row)
AREA = pd.DataFrame(area)
print(AREA.round(4).to_string(index=False))

# 효용 최적 승인 비율(각 점수별)
best = []
for nm in SCORES:
    m = C[C['점수'] == nm]
    best.append({'점수': nm, **{f'{k} 최적 승인비율': m.loc[m[k].idxmax(), '승인비율'] for k in ['시간 Sharpe', '예산 기준 초과수익', '확실성등가 γ=2', '확실성등가 γ=10', '확실성등가 γ=50']}})
BEST = pd.DataFrame(best)
print(BEST.to_string(index=False))

# ---------- C: 성과 귀속(등급 배분 vs 등급 안 선별) ----------
def brinson(a, label):
    bench = returns.pooled_irr(ev, fee=FEE)
    strat = returns.pooled_irr(ev[a], fee=FEE)
    rows, alloc, sel = [], 0.0, 0.0
    for g, ix in ev.groupby('grade').indices.items():
        wb = w[ix].sum() / w.sum()
        sa = ix[a[ix]]
        ws = w[sa].sum() / w[a].sum() if len(sa) else 0.0
        rb = returns.pooled_irr(ev.iloc[ix], fee=FEE)
        rs = returns.pooled_irr(ev.iloc[sa], fee=FEE) if len(sa) > 50 else rb
        alloc += (ws - wb) * (rb - bench)
        sel += ws * (rs - rb)
        rows.append((g, wb, ws, rb, rs))
    return {'규칙': label, '전부 승인 IRR': bench, '규칙 IRR': strat, '차이': strat - bench, '등급 배분 효과': alloc, '등급 안 선별 효과': sel,
            '잔차': strat - bench - alloc - sel}


BR = pd.DataFrame([brinson(top_share(SCORES['LightGBM 기대수익'], 0.5), 'LightGBM 기대수익 상위 50%'),
                   brinson(top_share(SCORES['LightGBM 부도확률(낮을수록)'], 0.3), 'LightGBM 부도확률 낮은 30%'),
                   brinson(top_share(SCORES['무작위(플라시보)'], 0.5), '무작위 50%')])
print(BR.round(4).to_string(index=False))

# ---------- D: 모델 없는 세부등급 규칙 ----------
D_rows = []
aD = np.zeros(len(ev), bool)
for Y in range(2013, 2017):
    C0 = pd.Timestamp(f'{Y}-01-01')
    hist = data.resolved_sample(full[(full['term_m'] == 36) & (full['ym'] <= C0 - pd.DateOffset(months=41)) & (full['ym'] >= C0 - pd.DateOffset(years=7))])
    hist = returns.attach_treasury(hist)
    rank = {}
    for sg, g in hist.groupby('sub_grade'):
        if len(g) >= 300:
            rank[sg] = returns.pooled_irr(g, fee=FEE) - g['tb'].mean()
    order = sorted(rank, key=rank.get, reverse=True)
    ix = np.where(yrs == Y)[0]
    sg_y = ev['sub_grade'].astype(str).to_numpy()[ix]
    chosen, cum = set(), 0.0
    for sg in order:  # 실현 초과수익이 높은 세부등급부터, Y년 대출의 50%가 찰 때까지
        if cum >= 0.5:
            break
        chosen.add(sg)
        cum += (sg_y == sg).mean()
    aD[ix] = np.isin(sg_y, list(chosen))
    D_rows.append({'연도': Y, '과거 표본(만기 도달)': len(hist), '고른 세부등급': ', '.join(sorted(chosen)), '승인비율': aD[ix].mean()})
print(pd.DataFrame(D_rows).to_string(index=False))
cmpD = []
for nm, a in {'세부등급 규칙(모델 없음)': aD, 'LightGBM 기대수익 상위 50%': top_share(SCORES['LightGBM 기대수익'], 0.5),
              '금리 낮은 50%': top_share(SCORES['금리(낮을수록)'], 0.5), '무작위 50%': top_share(SCORES['무작위(플라시보)'], 0.5)}.items():
    c = cohort_table(a)
    row = {'규칙': nm, '승인비율': a.mean(), '승인 대출 초과수익': (c['irr'] - c['tb']).mean(), '시간 Sharpe(예산 기준)': c['port_excess'].mean() / c['port_excess'].std(ddof=1)}
    for y in range(2013, 2017):
        row[f'{y}'] = (c.loc[c['year'] == y, 'irr'] - c.loc[c['year'] == y, 'tb']).mean()
    cmpD.append(row)
CD = pd.DataFrame(cmpD)
print(CD.round(4).to_string(index=False))

RESULTS.mkdir(exist_ok=True)
C.round(5).to_csv(RESULTS / '11_selection_curve.csv', index=False)
AREA.round(5).to_csv(RESULTS / '11_selection_value.csv', index=False)
BEST.to_csv(RESULTS / '11_utility_best_share.csv', index=False)
BR.round(5).to_csv(RESULTS / '11_brinson.csv', index=False)
CD.round(5).to_csv(RESULTS / '11_subgrade_rule.csv', index=False)

# C 보강: 연도별 성과 귀속(판정 기준 3)
ev_all = ev
rowsY = []
for Y in range(2013, 2017):
    m = yrs == Y
    ev = ev_all[m].reset_index(drop=True)
    w = ev['funded_amnt'].to_numpy()
    for nm, s in [('LightGBM 기대수익 상위 50%', SCORES['LightGBM 기대수익'][m]), ('무작위 50%', SCORES['무작위(플라시보)'][m])]:
        a = s >= np.quantile(s, 0.5)
        rowsY.append({'연도': Y, **brinson(a, nm)})
ev = ev_all
BRY = pd.DataFrame(rowsY)
print(BRY.round(4).to_string(index=False))
BRY.round(5).to_csv(RESULTS / '11_brinson_by_year.csv', index=False)
