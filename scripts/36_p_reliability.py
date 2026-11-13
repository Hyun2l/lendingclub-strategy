"""36단계(보고서용): p(부도확률)가 실제 부도율과 얼마나 맞는가.

시점 학습 LightGBM(시드 3개 평균)의 p를 10분위로 나눠 분위마다 평균 p와 실제 부도율을 비교한다(신뢰도 표).
같은 등급 안에서도 p가 부도를 가르는지(등급 안 분위표)와, 연도별 AUC·평균 p·실제 부도율도 함께 낸다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

y, p = ev['bad'].to_numpy(), ev['p'].to_numpy()
dec = pd.qcut(p, 10, labels=False)
R = pd.DataFrame({'분위': dec + 1, 'p': p, 'y': y, 'r': ev['r']}).groupby('분위').agg(
    대출수=('y', 'size'), 평균p=('p', 'mean'), 실제부도율=('y', 'mean'), 평균금리=('r', 'mean')).reset_index()
R.round(4).to_csv(RESULTS / '36_reliability.csv', index=False)

rows = []
for Y in range(2013, 2017):
    m = (ev['issue_year'] == Y).to_numpy()
    rows.append({'연도': Y, 'AUC': roc_auc_score(y[m], p[m]), '평균 p': p[m].mean(), '실제 부도율': y[m].mean(), '대출수': int(m.sum())})
Yt = pd.DataFrame(rows)
Yt.round(4).to_csv(RESULTS / '36_auc_by_year.csv', index=False)

g = ev['grade'].astype(str).to_numpy()
W = []
for gg in ['A', 'B', 'C', 'D', 'E']:
    m = g == gg
    q = pd.qcut(p[m], 3, labels=['낮음', '중간', '높음'])
    t = pd.DataFrame({'q': q, 'y': y[m], 'p': p[m]}).groupby('q', observed=True).agg(평균p=('p', 'mean'), 실제부도율=('y', 'mean'))
    W.append({'등급': gg, **{f'{k} p': v for k, v in t['평균p'].items()}, **{f'{k} 부도율': v for k, v in t['실제부도율'].items()},
              '등급 안 AUC': roc_auc_score(y[m], p[m])})
Wt = pd.DataFrame(W)
Wt.round(4).to_csv(RESULTS / '36_within_grade_reliability.csv', index=False)
pd.set_option('display.width', 220)
print(R.round(3).to_string(index=False))
print(Yt.round(3).to_string(index=False))
print(Wt.round(3).to_string(index=False))
