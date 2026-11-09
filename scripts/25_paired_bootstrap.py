"""25단계(검증): 23~24단계 비교의 차이가 잡음인지 발행월 단위 짝 부트스트랩으로 판정하고,
LightGBM에 가용성 규칙(학습 가용률 50% 미만 변수 제외)을 적용한 민감도 결과를 함께 본다.

방법:
- 각 모델(트리는 시드 0~2 평균 점수 곡선)에 대해 발행월마다 f_m = 승인비율 9개 평균의 (코호트 IRR - 국채) - 같은 달 무작위 승인의 값.
  연도 선별 가치는 f_m의 평균이므로, 발행월을 다시 뽑는 부트스트랩으로 차이의 불확실성을 잴 수 있다.
- 2013~2016년 48개 발행월을 짝지어(같은 달끼리) 1만 번 다시 뽑아 'LightGBM 기본'과의 차이 95% 구간을 구한다.
- 2017년(보류 구간, 이미 14단계에서 본 데이터, 종결 편향 있음)은 따로 보고만 한다.

판정 기준(사전 고정): Holm 보정 후 양측 p < 0.05이고 95% 구간이 0을 포함하지 않으면 '실제 차이'로 본다.
가용성 규칙 LightGBM은 test 프로토콜을 바꾸지 않는다. 민감도 보고용이다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])

import xgboost as xgb  # noqa: E402

B = 10000


def monthly(e, s):
    """발행월별 (IRR - 국채)를 승인비율 9개에 걸쳐 평균한 값."""
    return pd.concat([(lambda c: c['irr'] - c['tb'])(evaluate.cohort_irr(e, s >= np.quantile(s, 1 - q), fee=FEE)) for q in QS], axis=1).mean(axis=1)


months = []
for Y in range(2013, 2018):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    avail = X0.loc[tri, usable].notna().mean()
    usable_av = [c for c in usable if avail[c] >= 0.5]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    r, tb = ev['r'].to_numpy(), ev['tb'].to_numpy()
    rnd = pd.concat([monthly(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=1).mean(axis=1)
    Xt, Xe = X0.loc[tri, usable], X0.loc[evi, usable]
    Xc_t, Xc_e = Xt.copy(), Xe.copy()
    for c in [c for c in cats_all if c in usable]:
        Xc_t[c], Xc_e[c] = Xc_t[c].cat.codes.astype('float32'), Xc_e[c].cat.codes.astype('float32')
    P = {k: [] for k in ['LightGBM 기본', 'LightGBM 가용성 규칙', 'XGBoost', 'RandomForest']}
    for s in SEEDS:
        P['LightGBM 기본'].append(fit_lgb(Xt, ytr, s).predict_proba(Xe)[:, 1])
        P['LightGBM 가용성 규칙'].append(fit_lgb(Xt[usable_av], ytr, s).predict_proba(Xe[usable_av])[:, 1])
        u = np.random.default_rng(s).random(len(Xt)) < 0.15
        xm = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6,
                               min_child_weight=20, reg_lambda=5, tree_method='hist', enable_categorical=True, max_cat_to_onehot=1,
                               eval_metric='logloss', early_stopping_rounds=50, random_state=s, n_jobs=-1)
        xm.fit(Xt[~u], ytr[~u], eval_set=[(Xt[u], ytr[u])], verbose=False)
        P['XGBoost'].append(xm.predict_proba(Xe)[:, 1])
        P['RandomForest'].append(RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features='sqrt', max_samples=0.3,
                                                        n_jobs=-1, random_state=s).fit(Xc_t, ytr).predict_proba(Xc_e)[:, 1])
        print(Y, '시드', s, f'{(time.time() - t0) / 60:.1f}분', flush=True)
    C = choose_C(Xt.join(E.loc[tri]), ytr, 'l1')
    P['Lasso + 피처'] = [Linear('l1', C).fit(Xt.join(E.loc[tri]), ytr).predict(Xe.join(E.loc[evi]))]
    for nm, ps in P.items():
        f = pd.concat([monthly(ev, (1 - p) * r - p * lgd - tb) for p in ps], axis=1).mean(axis=1) - rnd
        months.append(pd.DataFrame({'연도': Y, '모델': nm, 'ym': f.index, 'f': f.to_numpy(),
                                    '평균 PD': float(np.mean([p.mean() for p in ps])), '실제 부도율': ev['bad'].mean()}))
    pd.concat(months).to_csv(RESULTS / '25_monthly_f.csv', index=False)
    print(Y, '완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)

M = pd.concat(months)
W = M.pivot_table(index=['연도', 'ym'], columns='모델', values='f')
main = W.loc[W.index.get_level_values('연도') <= 2016]
rng = np.random.default_rng(2026)
idx = rng.integers(0, len(main), size=(B, len(main)))
out = []
for m in [c for c in W.columns if c != 'LightGBM 기본']:
    diff = (main[m] - main['LightGBM 기본']).to_numpy()
    bs = diff[idx].mean(axis=1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    out.append({'모델': m, '차이(2013~16)': diff.mean(), '95% 하한': np.quantile(bs, 0.025), '95% 상한': np.quantile(bs, 0.975),
                'p': max(p, 1 / B), '이긴 달 비율': (diff > 0).mean(),
                '2017 차이(참고)': (W.loc[2017, m] - W.loc[2017, 'LightGBM 기본']).mean()})
S = pd.DataFrame(out).sort_values('p').reset_index(drop=True)
k = len(S)
S['Holm p'] = np.maximum.accumulate([min(1, p * (k - i)) for i, p in enumerate(S['p'])])
S['판정'] = np.where((S['Holm p'] < 0.05) & ((S['95% 하한'] > 0) | (S['95% 상한'] < 0)), '실제 차이', '잡음 안')
base = main['LightGBM 기본'].to_numpy()
bsb = base[idx].mean(axis=1)
print('LightGBM 기본 선별 가치(2013~16)', round(base.mean(), 5), '95% 구간', np.quantile(bsb, [0.025, 0.975]).round(5))
S.round(5).to_csv(RESULTS / '25_paired_bootstrap.csv', index=False)
pd.set_option('display.width', 220)
print(S.round(4).to_string(index=False))
print(M.groupby(['모델', '연도'])['f'].mean().unstack().round(4).to_string())
print(M.groupby(['모델', '연도'])[['평균 PD', '실제 부도율']].first().unstack().round(3).to_string())
print(f'총 {(time.time() - t0) / 60:.1f}분')
