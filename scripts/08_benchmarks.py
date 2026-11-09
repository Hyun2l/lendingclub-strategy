"""8단계: 비교 기준(거절한 돈을 넣을 곳)을 바꾸면 결론이 어떻게 달라지나.

07단계의 시점 학습 예측값을 그대로 쓰고, 비교 기준만 바꾼다.
- 승인 규칙 '기대수익 > 기준'은 기준이 높아질수록 덜 승인한다.
- 코호트 초과수익 = 승인 대출의 발행월별 IRR(수수료 1% 차감) - 그 달의 기준 수익률.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, evaluate, benchmarks
from lc.config import RESULTS, TRAIN_PARQUET

cols = ['loan_status', 'issue_d', 'int_rate', 'term', 'grade', 'last_pymnt_d', 'installment', 'total_pymnt',
        'funded_amnt', 'loan_amnt']
d = data.parse(data.load(columns=cols))
pred = pd.read_parquet(TRAIN_PARQUET.parent / '07_predictions.parquet')
ev = d.loc[pred.index].copy()
assert (ev['ym'].to_numpy() == pred['ym'].to_numpy()).all() and np.allclose(ev['r'], pred['r'])
er = {k: (1 - pred[f'p_{k}']) * pred['r'] - pred[f'p_{k}'] * pred['lgd'] for k in ['금리만', '로지스틱', 'LightGBM']}
pd30 = np.zeros(len(ev), bool)
p = pred['p_LightGBM'].to_numpy()
for _, ix in ev.reset_index(drop=True).groupby('issue_year').indices.items():
    pd30[ix] = p[ix] <= np.quantile(p[ix], 0.3)

rows = []
for b in benchmarks.NAMES:
    e = ev.copy()
    e['tb'] = benchmarks.benchmark(e, b)
    e = e.reset_index(drop=True)
    strat = {'전부 승인': np.ones(len(e), bool), 'LightGBM 부도확률 낮은 30%': pd30}
    for k, v in er.items():
        strat[f'{k} 기대수익 > 기준'] = v.to_numpy() > e['tb'].to_numpy()
    for nm, a in strat.items():
        s = evaluate.summarize(e, a)
        rows.append({'비교 기준': b, '기준 평균': e['tb'].mean(), '전략': nm, '승인비율': s['승인비율'],
                     '코호트 IRR': s['코호트_평균IRR'], '초과수익': s['코호트_초과수익'], '기준 이하 발행월(48개 중)': s['국채이하_코호트수'],
                     '코호트 Sharpe': s['코호트_Sharpe'], '대출별 Sharpe': s['대출별_Sharpe(국채재투자)'], '승인 부도율': s['승인_부도율']})
    print(b, '완료', flush=True)
t = pd.DataFrame(rows).round(4)
RESULTS.mkdir(exist_ok=True)
t.to_csv(RESULTS / '08_benchmarks.csv', index=False)
pd.set_option('display.width', 250)
print(t.to_string(index=False))
