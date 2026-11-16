"""37단계(외부 검증 대응): 승인 컷 단위(발행연도 vs 발행월)와 비교 기준(국채 vs BAA)에 따라 두 축 결론이 바뀌나.

외부 검증 요약의 지적:
(1) 표 7.4·7.6의 포트폴리오 Sharpe는 발행연도 단위로 상위 q를 잘랐다. 발행월 단위로 자르면 기대수익 순 Sharpe가 2.46 -> 2.88로 오른다.
(2) 비교 기준을 BAA로 올리면 기대수익 순이 수익·안정성 모두 1위가 된다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])
from lc import benchmarks  # noqa: E402

sharpe = lambda x: x.mean() / x.std(ddof=1)
BAA = np.asarray(benchmarks.benchmark(ev, '회사채 BAA'), float)


def top_by(score, q, key):
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby(key).indices.items():
        a[ix] = score[ix] >= np.quantile(score[ix], 1 - q)
    return a


def excess(a, hurdle):
    out = []
    for m in months:
        ix = grp[m]
        sel = ix[a[ix]]
        w = P_amt[sel].sum() / P_amt[ix].sum()
        out.append(w * (returns.pooled_irr(ev.iloc[sel], fee=FEE) - hurdle[ix].mean()) if len(sel) else 0.0)
    return np.array(out)


jit = np.random.default_rng(11).random(len(ev)) * 1e-6
rules = {'기대수익 순': ev['s_er'].to_numpy() + jit, '부도확률 순': -ev['p'].to_numpy() + jit}
rows = []
for key in ['issue_year', 'ym']:
    for q in (0.3, 0.5):
        for hname, h in [('국채', TB), ('BAA', BAA)]:
            for nm, s in rules.items():
                e = excess(top_by(s, q, key), h)
                rows.append({'컷 단위': key, '승인비율': q, '기준': hname, '규칙': nm, 'Sharpe': sharpe(e), '평균 초과(%p)': 100 * e.mean(),
                             '기준 이하 달': int((e <= 0).sum())})
            rs = [excess(top_by(np.random.default_rng(100 + k).random(len(ev)), q, key), h) for k in range(10)]
            rows.append({'컷 단위': key, '승인비율': q, '기준': hname, '규칙': '무작위(10회 평균)', 'Sharpe': np.mean([sharpe(r) for r in rs]),
                         '평균 초과(%p)': 100 * np.mean([r.mean() for r in rs]), '기준 이하 달': np.mean([(r <= 0).sum() for r in rs])})
T = pd.DataFrame(rows)
T.round(3).to_csv(RESULTS / '37_review_check.csv', index=False)
pd.set_option('display.width', 200)
print(T.round(2).to_string(index=False))
