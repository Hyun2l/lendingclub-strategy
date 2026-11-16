"""24b단계(진단): CatBoost 2016년 부도확률 수준 붕괴의 원인 점검.
학습 가용률 50% 미만 컬럼을 넣을 때와 뺄 때(가용성 규칙)를 같은 시드로 비교한다. 결과: results/24b_catboost_diag.log
"""
import _path  # noqa: F401
from pathlib import Path
SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])
from catboost import CatBoostClassifier
Y = 2016
tr = asof.training_set(d, f'{Y}-01-01'); tri, ytr = tr.index, tr['bad'].to_numpy()
usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True); lgd = asof.loss_given_default(tr)
rnd = np.mean([curve(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=0)
avail = X0.loc[tri, usable].notna().mean()
low = [c for c in usable if avail[c] < 0.5]
print('학습 가용률 50% 미만 컬럼', len(low))
# 가설 점검: 학습에서 '값이 있는' 대출과 '없는' 대출의 부도율
has = X0.loc[tri, low].notna().any(axis=1).to_numpy()
print('새 컬럼 있음: 부도율', ytr[has].mean().round(4), 'n', has.sum(), '| 없음:', ytr[~has].mean().round(4))
y = ev['bad'].to_numpy()
for name, cols in [('전체 컬럼', usable), ('가용성 규칙', [c for c in usable if avail[c] >= 0.5])]:
    cat_cols = [c for c in cats_all if c in cols]
    Xt, Xe = X0.loc[tri, cols].copy(), X0.loc[evi, cols].copy()
    for c in cat_cols:
        Xt[c], Xe[c] = Xt[c].astype(str).fillna('NA'), Xe[c].astype(str).fillna('NA')
    u = np.random.default_rng(0).random(len(Xt)) < 0.15
    cm = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=5, od_type='Iter', od_wait=50,
                            random_seed=0, verbose=0, thread_count=-1, cat_features=cat_cols)
    cm.fit(Xt[~u], ytr[~u], eval_set=(Xt[u], ytr[u]))
    p = cm.predict_proba(Xe)[:, 1]
    sv = float((curve(ev, (1 - p) * ev['r'].to_numpy() - p * lgd - ev['tb'].to_numpy()) - rnd).mean())
    print(name, '평균 PD', round(p.mean(), 4), 'AUC', round(roc_auc_score(y, p), 4), 'Brier', round(brier_score_loss(y, p), 4), 'SV', round(sv, 4))
