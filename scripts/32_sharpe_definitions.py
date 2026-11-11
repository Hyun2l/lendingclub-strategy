"""32단계(검증): Sharpe 정의 두 가지의 실제 크기, 그리고 검증 구간에서 Sharpe로 승인비율을 고르는 규칙.

배경: 조교 "Sharpe 0.2~0.4면 옳게 한 것", 교수 "부도확률 몇 이상이면 거절할지 Sharpe 극대화로 정한다".
(a) 대출 건별 Sharpe(과제식): 대출마다 x = 승인이면 (그 대출 수익률 - 국채), 거절이면 0. 전체 대출에 걸친 평균 ÷ 표준편차.
    대출 수익률은 받은 돈을 국채로 만기까지 재투자한 연수익률(수수료 1%). 건별 IRR은 극단값이 커서 쓰지 않는다.
    승인한 대출만의 건별 Sharpe도 함께 본다.
(b) 발행월별 포트폴리오 Sharpe(30단계 정의, IRR 기준).
(c) 승인비율 선택: 2013~14년(검증)에서 포트폴리오 Sharpe가 최대인 승인비율을 고르고, 2015~16년(평가)에 그대로 적용한다.
    주의: 2014년 대출의 결과는 2015년 초에 다 알 수 없으므로 실제보다 유리한 '사후 검증'이다. 방향 확인용.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

sharpe = lambda x: x.mean() / x.std(ddof=1)
R_loan = (ev['fv'] / ev['funded_amnt']) ** (12 / 36) - 1
x_loan = (R_loan - ev['tb']).to_numpy()
rules = {'기대수익 순': ev['s_er'].to_numpy(), '부도확률 낮은 순': -ev['p'].to_numpy(), '금리 낮은 순': -ev['r'].to_numpy()}
QQ = QS + [1.0]

rows = []
for nm, s in list(rules.items()) + [('무작위', None)]:
    for q in QQ:
        draws = [top(np.random.default_rng(100 + k).random(len(ev)), q) for k in range(5)] if s is None else [top(s, q)]
        ls, la, ps, pm = [], [], [], []
        for a in draws:
            ls.append(sharpe(np.where(a, x_loan, 0.0)))
            la.append(sharpe(x_loan[a]))
            ex = monthly_excess(a)[0]
            ps.append(sharpe(ex))
            pm.append(ex)
        pm = np.mean(pm, axis=0)
        rows.append({'전략': nm, '승인비율': q, '건별 Sharpe(거절=0)': np.mean(ls), '건별 Sharpe(승인만)': np.mean(la),
                     '포트폴리오 Sharpe': np.mean(ps), '_monthly': pm})
T = pd.DataFrame(rows)

val = np.array([str(m)[:4] in ('2013', '2014') for m in months])
pick = []
for nm in list(rules) + ['무작위']:
    t = T[T['전략'] == nm]
    sv = t['_monthly'].apply(lambda m: sharpe(m[val]))
    best = t.loc[sv.idxmax()]
    m = best['_monthly']
    pick.append({'전략': nm, '검증에서 고른 승인비율': best['승인비율'], '검증 Sharpe': sharpe(m[val]),
                 '평가(2015~16) Sharpe': sharpe(m[~val]), '평가 평균 초과수익(%p)': 100 * m[~val].mean()})
allm = T[(T['전략'] == '무작위') & (T['승인비율'] == 1.0)]['_monthly'].iloc[0]
pick.append({'전략': '전부 승인', '검증에서 고른 승인비율': 1.0, '검증 Sharpe': sharpe(allm[val]),
             '평가(2015~16) Sharpe': sharpe(allm[~val]), '평가 평균 초과수익(%p)': 100 * allm[~val].mean()})
P = pd.DataFrame(pick)

T.drop(columns='_monthly').round(4).to_csv(RESULTS / '32_sharpe_definitions.csv', index=False)
P.round(4).to_csv(RESULTS / '32_threshold_by_sharpe.csv', index=False)
pd.set_option('display.width', 220)
for col in ['건별 Sharpe(거절=0)', '건별 Sharpe(승인만)', '포트폴리오 Sharpe']:
    print(f'\n== {col}')
    print(T.pivot(index='승인비율', columns='전략', values=col).round(3).to_string())
print('\n== 검증(2013~14)에서 Sharpe 최대 승인비율을 골라 2015~16에 적용')
print(P.round(3).to_string(index=False))
print(f'총 {(time.time() - t0) / 60:.1f}분')
