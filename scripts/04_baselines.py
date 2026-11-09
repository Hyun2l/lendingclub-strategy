"""4단계: 기준선 전략 표.

평가 표본: 2013~2016년 발행 36개월 종결 대출(이 구간은 99.9% 이상 종결되어 결과 편향이 거의 없다).
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, returns, evaluate, baselines
from lc.config import RESULTS

cols = ['loan_status', 'issue_d', 'int_rate', 'term', 'grade', 'last_pymnt_d', 'installment', 'total_pymnt',
        'funded_amnt', 'loan_amnt']
d = data.parse(data.load(columns=cols))
print('평가 구간 종결 비율', round(d[d.issue_year.between(2013, 2016) & (d.term_m == 36)].resolved.mean(), 5))
d = data.resolved_sample(d)
d = returns.attach_treasury(d[d.issue_year.between(2013, 2016) & (d.term_m == 36)].reset_index(drop=True))
print('평가 표본', len(d), '발행월', d.ym.nunique())

rows = {nm: evaluate.summarize(d, a) for nm, a in baselines.baselines(d).items()}
t = pd.DataFrame(rows).T.round(4)
RESULTS.mkdir(exist_ok=True)
t.to_csv(RESULTS / '04_baselines.csv')
print(t.to_string())

# 무작위 승인은 선별 실력이 없으므로 코호트 IRR이 전부 승인과 같아야 한다(플라시보).
rand = [evaluate.summarize(d, np.random.default_rng(s).random(len(d)) < 0.3)['코호트_평균IRR'] for s in range(1, 6)]
print('무작위 30% 다섯 번의 코호트 평균IRR', np.round(rand, 4), '| 전부 승인', round(t.loc['전부 승인', '코호트_평균IRR'], 4))
assert abs(np.mean(rand) - t.loc['전부 승인', '코호트_평균IRR']) < 0.002
