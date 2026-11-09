"""2단계 점검: 표본 수, 결과 상태 분포, 누수 컬럼이 특성에 없는지."""
import _path  # noqa: F401
import pandas as pd
from lc import data
from lc.config import RESULTS

d = data.parse(data.load())
feats = data.feature_columns(d)
res = data.resolved_sample(d)
leaks = [c for c in feats if data.is_leak(c)]
print('전체', len(d), '| 종결 표본', len(res), '| 특성', len(feats), '| 특성 속 누수', leaks)
assert not leaks and 'loan_status' not in feats and 'total_pymnt' not in feats
assert {'r', 'term_m', 'grade', 'fico_range_low'} <= set(feats)

tab = pd.crosstab(d['issue_year'], d['outcome'])[list(data.OUTCOMES)]
tab['종결비율'] = (tab['paid'] + tab['default']) / tab.sum(axis=1)
tab['종결중_부도율'] = tab['default'] / (tab['paid'] + tab['default'])
RESULTS.mkdir(exist_ok=True)
tab.round(4).to_csv(RESULTS / '02_status_by_year.csv')
print(tab.round(3).to_string())
pd.Series(feats, name='feature').to_csv(RESULTS / '02_features.csv', index=False)
