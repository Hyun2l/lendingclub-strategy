"""29단계(민감도): 대출액 비례 투자 대신 '대출마다 같은 금액' 투자로 채점해도 결과가 같은가.

배경: 지금까지의 코호트 IRR은 대출액에 비례해 투자한 가상의 투자자다. 실제 개인 투자자는 대출마다
같은 금액(예: 25달러)씩 나눠 사는 경우가 많다. 같은 금액 방식은 각 대출의 현금흐름을 대출액으로 나눠(1달러 투자) 더한다.

조건: 25단계와 같다(시점 학습 2013~16, 36개월, LightGBM 기본 시드 3개, 무작위 5회, 발행월 부트스트랩 1만 번).
판정(사전 고정): 같은 금액 방식의 선별 가치 95% 구간 하한이 0보다 크면 '결론 유지'.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])


def monthly(e, s, equal):
    """발행월별 (승인비율 9개 평균의 IRR - 국채). equal=True면 대출마다 1달러 투자."""
    w_all = 1.0 / e['funded_amnt'].to_numpy(float) if equal else np.ones(len(e))
    out = {}
    for ym, ix in e.groupby('ym').indices.items():
        tb = e['tb'].to_numpy()[ix].mean()
        vals = []
        for q in QS:
            a = ix[s[ix] >= np.quantile(s, 1 - q)]
            vals.append((returns.pooled_irr(e.iloc[a], weight=w_all[a], fee=FEE) if len(a) else tb) - tb)
        out[ym] = np.mean(vals)
    return pd.Series(out)


rows = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    r, tb = ev['r'].to_numpy(), ev['tb'].to_numpy()
    P = [fit_lgb(X0.loc[tri, usable], ytr, s).predict_proba(X0.loc[evi, usable])[:, 1] for s in SEEDS]
    ones = np.ones(len(ev))
    for nm, eq in [('대출액 비례', False), ('대출마다 같은 금액', True)]:
        rnd = pd.concat([monthly(ev, np.random.default_rng(100 + k).random(len(ev)), eq) for k in range(5)], axis=1).mean(axis=1)
        f = pd.concat([monthly(ev, (1 - p) * r - p * lgd - tb, eq) for p in P], axis=1).mean(axis=1) - rnd
        w = 1.0 / ev['funded_amnt'].to_numpy(float) if eq else ones
        allap = pd.Series({ym: returns.pooled_irr(ev.iloc[ix], weight=w[ix], fee=FEE) - ev['tb'].to_numpy()[ix].mean()
                           for ym, ix in ev.groupby('ym').indices.items()})
        rows.append(pd.DataFrame({'연도': Y, '방식': nm, 'ym': f.index, '선별 가치': f.to_numpy(), '전부 승인 초과수익': allap.loc[f.index].to_numpy()}))
    print(Y, f'{(time.time() - t0) / 60:.1f}분', flush=True)

M = pd.concat(rows)
rng = np.random.default_rng(2026)
summ = []
for nm, g in M.groupby('방식', sort=False):
    v = g['선별 가치'].to_numpy()
    bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
    summ.append({'방식': nm, '선별 가치': v.mean(), '95% 하한': np.quantile(bs, 0.025), '95% 상한': np.quantile(bs, 0.975),
                 '전부 승인 초과수익': g['전부 승인 초과수익'].mean(), '국채 이하 발행월 수(전부 승인)': int((g['전부 승인 초과수익'] <= 0).sum())})
S = pd.DataFrame(summ)
S.round(5).to_csv(RESULTS / '29_equal_weight.csv', index=False)
pd.set_option('display.width', 200)
print(S.round(4).to_string(index=False))
print(M.groupby(['방식', '연도'])['선별 가치'].mean().unstack().round(4).to_string())
eqrow = S.set_index('방식').loc['대출마다 같은 금액']
print('판정:', '결론 유지' if eqrow['95% 하한'] > 0 else '결론 약화')
