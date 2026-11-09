"""3단계: 발행연도별로 수익률 정의를 비교한다. 같은 대출도 정의에 따라 국채보다 높게도, 낮게도 나온다."""
import _path  # noqa: F401
import pandas as pd
from lc import data, returns
from lc.config import RESULTS

cols = ['loan_status', 'issue_d', 'int_rate', 'term', 'last_pymnt_d', 'installment', 'total_pymnt', 'funded_amnt', 'loan_amnt']
d = returns.attach_treasury(data.resolved_sample(data.parse(data.load(columns=cols))))
d['수업식'] = returns.class_return(d)
d['단순연환산'] = returns.simple_annualized(d)
d['국채재투자'] = returns.reinvested_return(d, d['tb'])
rows = []
for y, g in d[d.issue_year.between(2010, 2018)].groupby('issue_year'):
    rows.append({'발행연도': y, '건수': len(g), '부도율': g.bad.mean(), '약정금리': g.r.mean(),
                 '수업식': g['수업식'].mean(), '단순연환산': g['단순연환산'].mean(), '국채재투자': g['국채재투자'].mean(),
                 '통합IRR': returns.pooled_irr(g), '통합IRR_수수료후': returns.pooled_irr(g, fee=0.01), '국채': g.tb.mean()})
t = pd.DataFrame(rows).round(4)
RESULTS.mkdir(exist_ok=True)
t.to_csv(RESULTS / '03_returns_by_year.csv', index=False)
print(t.to_string(index=False))
print('※ 2017년 이후는 종결 대출만 남은 편향 때문에 낮게 나온다.')
