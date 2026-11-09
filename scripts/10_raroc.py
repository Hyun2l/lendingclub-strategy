"""10단계: 같은 예측값(7단계 시점 학습 LightGBM)으로 세 관점의 승인 규칙을 비교한다.

- 개인 투자자(Sharpe 계열): 기대수익 > 만기 국채
- 기관 투자자(정보비율 계열): 기대수익 > 회사채 BAA
- 은행(RAROC): RAROC > 요구수익률 (기본 가정: 조달 = 만기 국채 + 1%p, 운영비 1.5%, 요구수익률 12%, 자본 = 바젤 K)
실현 성과는 은행 관점이라 투자자 수수료를 빼지 않은 발행월 코호트 IRR로 잰다(2013~2016년 36개월).
"""
import itertools
import _path  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lc import data, evaluate, benchmarks, raroc
from lc.config import RESULTS, TRAIN_PARQUET

cols = ['loan_status', 'issue_d', 'int_rate', 'term', 'grade', 'last_pymnt_d', 'installment', 'total_pymnt',
        'funded_amnt', 'loan_amnt']
d = data.parse(data.load(columns=cols))
pred = pd.read_parquet(TRAIN_PARQUET.parent / '07_predictions.parquet')
ev = d.loc[pred.index].copy()
ev['tb'] = benchmarks.benchmark(ev, '만기 국채')
ev['baa'] = benchmarks.benchmark(ev, '회사채 BAA')
ev = ev.reset_index(drop=True)
p = pred['p_LightGBM'].to_numpy()
lgd = pred['lgd'].to_numpy()
r = ev['r'].to_numpy()
T = ev['term_m'].to_numpy()
er = (1 - p) * r - p * lgd
cohorts = ev.groupby('ym').indices


def bank_metrics(a, spread_s, opex, h, cap_mult):
    """발행월마다: 실현 IRR(수수료 없음) - 조달 - 운영비 = 실현 순마진, 자본 = 승인 대출의 금액가중 평균 K."""
    _, cap = raroc.components(r, p, T, lgd, ev['tb'].to_numpy() + spread_s, opex, cap_mult)
    out = []
    w = ev['funded_amnt'].to_numpy()
    for ym, ix in cohorts.items():
        s = ix[a[ix]]
        if len(s) == 0:
            continue
        from lc import returns
        irr = returns.pooled_irr(ev.iloc[s], fee=0.0)
        f = np.average(ev['tb'].to_numpy()[s] + spread_s, weights=w[s])
        k = np.average(cap[s], weights=w[s])
        margin = irr - f - opex
        out.append((margin, k, margin / k, (margin - h * k) * w[s].sum(), w[s].sum()))
    m = np.array(out)
    return {'실현 순마진': m[:, 0].mean(), '평균 자본비율': m[:, 1].mean(), '실현 RAROC': m[:, 2].mean(),
            '실현 RAROC 최저 발행월': m[:, 2].min(), '경제적 이익 합계($M)': m[:, 3].sum() / 1e6, '투자 금액($M)': m[:, 4].sum() / 1e6}


BASE = dict(spread_s=0.01, opex=0.015, h=0.12, cap_mult=1.0)
sc = raroc.raroc(r, p, T, lgd, ev['tb'].to_numpy() + BASE['spread_s'], BASE['opex'], BASE['cap_mult'])
rules = {'전부 승인': np.ones(len(ev), bool), '기대수익 > 국채(Sharpe 계열)': er > ev['tb'].to_numpy(),
         '기대수익 > BAA(정보비율 계열)': er > ev['baa'].to_numpy(), 'RAROC > 12%(은행)': sc > BASE['h']}
rows = []
for nm, a in rules.items():
    s = evaluate.summarize(ev, a, fee=0.0)
    ex_baa = evaluate.cohort_irr(ev.assign(tb=ev['baa']), a, fee=0.0)
    ir = (ex_baa['irr'] - ex_baa['tb'])
    rows.append({'규칙': nm, '승인비율': a.mean(), '코호트 IRR': s['코호트_평균IRR'], '국채 대비 코호트 Sharpe': s['코호트_Sharpe'],
                 '과제식 Sharpe': s['과제식_Sharpe(부도=0, rf5%)'], 'BAA 대비 정보비율': ir.mean() / ir.std(ddof=1),
                 '승인 부도율': s['승인_부도율'], **bank_metrics(a, **BASE)})
    print(nm, '완료', flush=True)
cmp_ = pd.DataFrame(rows)
pd.set_option('display.width', 260)
print(cmp_.round(4).to_string(index=False))

# 규칙끼리 얼마나 다른가
A = rules['RAROC > 12%(은행)']; B = rules['기대수익 > 국채(Sharpe 계열)']; Cc = rules['기대수익 > BAA(정보비율 계열)']
jac = lambda x, y: (x & y).sum() / (x | y).sum()
print('RAROC 점수와 (기대수익-국채)의 순위상관', round(spearmanr(sc, er - ev['tb'])[0], 3),
      '| 승인 집합 겹침(자카드) RAROC~국채', round(jac(A, B), 3), 'RAROC~BAA', round(jac(A, Cc), 3))
print('RAROC 분포(분위수 5/25/50/75/95%)', np.round(np.quantile(sc, [.05, .25, .5, .75, .95]), 3),
      '| 바젤 K 중앙값', round(float(np.median(raroc.basel_retail_k(raroc.annual_pd(p, T), lgd))), 4))

# 가정 민감도: 가정이 바뀌면 승인 결정과 성과가 얼마나 흔들리나
sens = []
for s_, o, h, cm in itertools.product([0.0, 0.01, 0.02], [0.01, 0.015, 0.025], [0.08, 0.12, 0.15], [1.0, 2.0]):
    a = raroc.raroc(r, p, T, lgd, ev['tb'].to_numpy() + s_, o, cm) > h
    bm = bank_metrics(a, s_, o, h, cm)
    sens.append({'조달 가산': s_, '운영비': o, '요구수익률': h, '자본 배수': cm, '승인비율': a.mean(),
                 '실현 RAROC': bm['실현 RAROC'], '경제적 이익 합계($M)': bm['경제적 이익 합계($M)'],
                 '전부 승인 경제적 이익($M)': bank_metrics(np.ones(len(ev), bool), s_, o, h, cm)['경제적 이익 합계($M)']})
S = pd.DataFrame(sens)
S['선별 효과($M)'] = S['경제적 이익 합계($M)'] - S['전부 승인 경제적 이익($M)']
print(S.round(4).to_string(index=False))
RESULTS.mkdir(exist_ok=True)
cmp_.round(4).to_csv(RESULTS / '10_raroc_compare.csv', index=False)
S.round(4).to_csv(RESULTS / '10_raroc_sensitivity.csv', index=False)
