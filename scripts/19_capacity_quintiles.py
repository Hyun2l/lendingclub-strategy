"""19단계(탐색, 모델 없음): 같은 세부등급 안에서 상환 능력(연소득, 부채비율) 5분위별 실현 IRR.
기준(미리 정함): 2013~2016년 4개 연도 모두에서 '소득 상위 20% IRR > 하위 20%', '부채비율 하위 20% IRR > 상위 20%'이면
'LC 등급이 상환 능력을 덜 반영했다'는 가설을 지지한다."""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, returns
from lc.config import RESULTS

FEE = 0.01
d = data.parse(data.load())
e = returns.attach_treasury(data.resolved_sample(d)[lambda x: (x['term_m'] == 36) & x['issue_year'].between(2013, 2016)])
for col, lab in [('annual_inc', 'inc_q'), ('dti', 'dti_q')]:
    e[lab] = e.groupby(['issue_year', 'sub_grade'], observed=True)[col].transform(
        lambda s: np.ceil(s.rank(method="first", pct=True) * 5).clip(1, 5))
rows = []
for col, lab, name in [('annual_inc', 'inc_q', '연소득'), ('dti', 'dti_q', '부채비율')]:
    for y in ['통합', 2013, 2014, 2015, 2016]:
        g = e if y == '통합' else e[e['issue_year'] == y]
        r = {'변수': name, '연도': y}
        for q in range(1, 6):
            s = g[g[lab] == q]
            r[f'Q{q}'] = returns.pooled_irr(s, fee=FEE)
        r['Q5-Q1'] = r['Q5'] - r['Q1']
        r['Q1 부도율'] = g.loc[g[lab] == 1, 'bad'].mean()
        r['Q5 부도율'] = g.loc[g[lab] == 5, 'bad'].mean()
        r['Q1 금리'] = g.loc[g[lab] == 1, 'r'].mean()
        r['Q5 금리'] = g.loc[g[lab] == 5, 'r'].mean()
        rows.append(r)
T = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
T.round(5).to_csv(RESULTS / '19_capacity_quintiles.csv', index=False)
pd.set_option('display.width', 220)
print(T.round(4).to_string(index=False))
yr = T[T['연도'] != '통합']
inc_ok = (yr[yr['변수'] == '연소득']['Q5-Q1'] > 0).sum()
dti_ok = (yr[yr['변수'] == '부채비율']['Q5-Q1'] < 0).sum()
print(f'소득 상위>하위인 연도 {inc_ok}/4, 부채비율 하위>상위인 연도 {dti_ok}/4 -> {"지지" if inc_ok == 4 and dti_ok == 4 else "지지 안 됨"}')
