"""24단계(탐색): XGBoost, CatBoost, Lasso + 피처를 23단계와 같은 조건으로 추가 비교한다.
23단계의 전처리·피처·평가 함수를 그대로 쓴다(같은 파일에서 읽어 실행). 기준 모델 LightGBM 기본도 같은 시드로 다시 돌려 짝 비교한다.
판정 기준(23단계와 같음): 'LightGBM 기본'보다 선별 가치 0.1%p 초과 또는 AUC 0.002 초과로 나은 연도가 3개 이상이면 도전자.
Lasso + 피처는 'Lasso(C 선택)'(23단계 선형 채택안)과 같은 기준으로 비교한다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])  # 데이터, 피처, Linear, choose_C, curve, within, fit_lgb

import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

rows = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    rnd = np.mean([curve(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=0)
    Xt, Xe = X0.loc[tri, usable], X0.loc[evi, usable]
    cat_cols = [c for c in cats_all if c in usable]
    Xt_cb, Xe_cb = Xt.copy(), Xe.copy()
    for c in cat_cols:
        Xt_cb[c], Xe_cb[c] = Xt_cb[c].astype(str).fillna('NA'), Xe_cb[c].astype(str).fillna('NA')
    preds = {}
    C = choose_C(Xt.join(E.loc[tri]), ytr, 'l1')
    preds[('Lasso(C 선택)', 0)] = Linear('l1', choose_C(Xt, ytr, 'l1')).fit(Xt, ytr).predict(Xe)
    preds[('Lasso + 피처', 0)] = Linear('l1', C).fit(Xt.join(E.loc[tri]), ytr).predict(Xe.join(E.loc[evi]))
    print(Y, 'Lasso 완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)
    for s in SEEDS:
        preds[('LightGBM 기본', s)] = fit_lgb(Xt, ytr, s).predict_proba(Xe)[:, 1]
        u = np.random.default_rng(s).random(len(Xt)) < 0.15
        xm = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6,
                               min_child_weight=20, reg_lambda=5, tree_method='hist', enable_categorical=True, max_cat_to_onehot=1,
                               eval_metric='logloss', early_stopping_rounds=50, random_state=s, n_jobs=-1)
        xm.fit(Xt[~u], ytr[~u], eval_set=[(Xt[u], ytr[u])], verbose=False)
        preds[('XGBoost', s)] = xm.predict_proba(Xe)[:, 1]
        cm = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=5, od_type='Iter', od_wait=50,
                                random_seed=s, verbose=0, thread_count=-1, cat_features=cat_cols)
        cm.fit(Xt_cb[~u], ytr[~u], eval_set=(Xt_cb[u], ytr[u]))
        preds[('CatBoost', s)] = cm.predict_proba(Xe_cb)[:, 1]
        print(Y, '시드', s, f'{(time.time() - t0) / 60:.1f}분', flush=True)
    for (nm, s), p in preds.items():
        score = (1 - p) * ev['r'].to_numpy() - p * lgd - ev['tb'].to_numpy()
        cv = curve(ev, score)
        rows.append({'연도': Y, '모델': nm, '시드': s, 'AUC': roc_auc_score(ev['bad'], p), 'Brier': brier_score_loss(ev['bad'], p),
                     '선별 가치': float((cv - rnd).mean()), '등급 안 선별': within(ev, score >= np.quantile(score, 0.5))})
    pd.DataFrame(rows).round(5).to_csv(RESULTS / '24_boosting_rows.csv', index=False)

R = pd.DataFrame(rows)
Yr = R.groupby(['모델', '연도'])[['AUC', '선별 가치', '등급 안 선별', 'Brier']].mean()


def compare(base_name, names):
    base = Yr.xs(base_name, level='모델')
    out = []
    for m in names:
        t = Yr.xs(m, level='모델')
        out.append({'모델': m, '기준': base_name, 'AUC': t['AUC'].mean(), '선별 가치': t['선별 가치'].mean(), '등급 안 선별': t['등급 안 선별'].mean(),
                    'Brier': t['Brier'].mean(),
                    '나은 연도': int(((t['선별 가치'] - base['선별 가치'] > 0.001) | (t['AUC'] - base['AUC'] > 0.002)).sum()),
                    '나쁜 연도': int(((base['선별 가치'] - t['선별 가치'] > 0.001) | (base['AUC'] - t['AUC'] > 0.002)).sum())})
    return pd.DataFrame(out)


S = pd.concat([compare('LightGBM 기본', ['LightGBM 기본', 'XGBoost', 'CatBoost']), compare('Lasso(C 선택)', ['Lasso + 피처'])])
S.round(5).to_csv(RESULTS / '24_boosting_summary.csv', index=False)
Yr.round(5).to_csv(RESULTS / '24_boosting_by_year.csv')
pd.set_option('display.width', 220)
print(S.round(4).to_string(index=False))
print(Yr['선별 가치'].unstack().round(4).to_string())
print(Yr['AUC'].unstack().round(4).to_string())
print(Yr['Brier'].unstack().round(4).to_string())
print(f'총 {(time.time() - t0) / 60:.1f}분')
