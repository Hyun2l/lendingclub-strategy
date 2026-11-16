"""28단계(점검): funded_amnt_inv(투자자가 실제로 넣은 금액)를 빼도 결과가 같은가.

배경: 지난 팀 발표(4조)의 시점 구분에서 이 컬럼은 투자 모집이 끝난 뒤에야 확정되는 정보(t_inv)다.
투자자가 대출을 고르는 순간에는 알 수 없으므로 약한 누수 후보다. 기본 101개 특성에 들어 있다(중요도 48위).
test 프로토콜(v1)은 바꾸지 않고, 민감도 결과로만 기록한다.

방법: 25단계와 같은 시점 학습 2013~16, LightGBM 시드 3개, 발행월 짝 부트스트랩 1만 번.
판정(사전 고정): 차이의 95% 구간이 0을 포함하면 '결과에 영향 없음'.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])


def monthly(e, s):
    return pd.concat([(lambda c: c['irr'] - c['tb'])(evaluate.cohort_irr(e, s >= np.quantile(s, 1 - q), fee=FEE)) for q in QS], axis=1).mean(axis=1)


rows = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    drop = [c for c in usable if c != 'funded_amnt_inv']
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    r, tb = ev['r'].to_numpy(), ev['tb'].to_numpy()
    rnd = pd.concat([monthly(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=1).mean(axis=1)
    for nm, cols in [('기본 101개', usable), ('funded_amnt_inv 제외', drop)]:
        P = [fit_lgb(X0.loc[tri, cols], ytr, s).predict_proba(X0.loc[evi, cols])[:, 1] for s in SEEDS]
        f = pd.concat([monthly(ev, (1 - p) * r - p * lgd - tb) for p in P], axis=1).mean(axis=1) - rnd
        auc = np.mean([roc_auc_score(ev['bad'], p) for p in P])
        rows.append(pd.DataFrame({'연도': Y, '모델': nm, 'ym': f.index, 'f': f.to_numpy(), 'AUC': auc}))
    print(Y, f'{(time.time() - t0) / 60:.1f}분', flush=True)

M = pd.concat(rows)
W = M.pivot_table(index=['연도', 'ym'], columns='모델', values='f')
diff = (W['funded_amnt_inv 제외'] - W['기본 101개']).to_numpy()
bs = diff[np.random.default_rng(2026).integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
lo, hi = np.quantile(bs, [0.025, 0.975])
out = M.groupby(['모델', '연도'])[['f', 'AUC']].mean().unstack().round(4)
out.to_csv(RESULTS / '28_drop_funded_inv.csv')
print(out.to_string())
print(f'차이(제외 - 기본) {diff.mean():+.5f}, 95% 구간 [{lo:+.5f}, {hi:+.5f}]')
print('판정:', '결과에 영향 없음' if lo <= 0 <= hi else '실제 차이 있음')
