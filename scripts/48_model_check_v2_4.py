"""48단계(보고만): v2.4 선택이 특정 모델(LightGBM) 덕인지 확인한다. 선택안(PICK)은 바꾸지 않는다.

v2.4는 p의 크기가 아니라 그달 안의 p 순서로 고른다. 그래서 순서를 잘 매기는 모델이 더 나을 수 있다.
같은 시점 학습(2013~16, 36개월 종결 대출, 그해 1월 1일까지 확정된 대출로 학습)으로 모델 다섯 개의 p를 만들고,
46단계와 같은 건별 Sharpe(대출별 IRR, 약정 IRR 상한, 수수료 1%, 거절 = 0)를 V1·V2·V4 × 승인 10~90%로 계산한다.
모델: LightGBM 기본(46단계와 같음), XGBoost(24단계 설정), RandomForest·HistGradientBoosting(23단계 설정), L2 로지스틱(23단계 'L2 기준').
트리 모델은 시드 3개 평균 p. 불확실성: 발행월 6개월 블록 부트스트랩 2000회(seed 2026), 같은 인덱스로 LightGBM과 짝지은 차이.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '23_models_final.py').read_text()
exec(SRC[:SRC.index('LGB_GRID = ')])
import xgboost as xgb  # noqa: E402

B, BLOCK = 2000, 6
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def fit_xgb(Xt, yt, seed):
    u = np.random.default_rng(seed).random(len(Xt)) < 0.15
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.6,
                          min_child_weight=20, reg_lambda=5, tree_method='hist', enable_categorical=True, max_cat_to_onehot=1,
                          eval_metric='logloss', early_stopping_rounds=50, random_state=seed, n_jobs=-1)
    m.fit(Xt[~u], yt[~u], eval_set=[(Xt[u], yt[u])], verbose=False)
    return m


MODELS = ['LightGBM 기본', 'XGBoost', 'RandomForest', 'HistGradientBoosting', 'L2 로지스틱']
EV, AUC = [], []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    ev['lgd'] = asof.loss_given_default(tr)
    Xt, Xe = X0.loc[tri, usable], X0.loc[evi, usable]
    Xt_c, Xe_c = Xt.copy(), Xe.copy()
    for c in [c for c in cats_all if c in usable]:
        Xt_c[c], Xe_c[c] = Xt_c[c].cat.codes.astype('float32'), Xe_c[c].cat.codes.astype('float32')
    P = {m: [] for m in MODELS}
    for s in SEEDS:
        P['LightGBM 기본'].append(fit_lgb(Xt, ytr, s).predict_proba(Xe)[:, 1])
        P['XGBoost'].append(fit_xgb(Xt, ytr, s).predict_proba(Xe)[:, 1])
        P['RandomForest'].append(RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features='sqrt', max_samples=0.3,
                                                        n_jobs=-1, random_state=s).fit(Xt_c, ytr).predict_proba(Xe_c)[:, 1])
        P['HistGradientBoosting'].append(HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True,
                                                                        categorical_features='from_dtype', random_state=s)
                                         .fit(Xt, ytr).predict_proba(Xe)[:, 1])
    P['L2 로지스틱'].append(Linear('l2', 1.0).fit(Xt, ytr).predict(Xe))
    for m in MODELS:
        ev[f'p|{m}'] = np.mean(P[m], axis=0)
        AUC.append({'연도': Y, '모델': m, 'AUC': roc_auc_score(ev['bad'], ev[f'p|{m}']), '평균 p': ev[f'p|{m}'].mean(), '실제 부도율': ev['bad'].mean()})
    EV.append(ev)
    log(f'{Y}: 학습 {len(tr):,}건, 평가 {len(ev):,}건, {(time.time() - t0) / 60:.1f}분')
ev = pd.concat(EV, ignore_index=True)
n = len(ev)
TB = ev['tb'].to_numpy(float)
X = np.minimum(returns.loan_irr(ev, fee=FEE).to_numpy(), (1 + ev['r'].to_numpy() / 12) ** 12 - 1) - TB

months = np.array(sorted(ev['ym'].unique()))
grp = ev.groupby('ym').indices
MI = np.zeros(n, int)
for k, m in enumerate(months):
    MI[grp[m]] = k
NM = len(months)
st = np.random.default_rng(2026).integers(0, NM - BLOCK + 1, size=(B, int(np.ceil(NM / BLOCK))))
BIX = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :NM], NM - 1)


def by_month_top(score, q):
    a = np.zeros(n, bool)
    for ix in grp.values():
        a[ix] = score[ix] >= np.quantile(score[ix], 1 - q)
    return a


def boot(a):
    x = np.where(a, X, 0.0)
    S1, S2, N = np.bincount(MI, x, NM), np.bincount(MI, x * x, NM), np.bincount(MI, minlength=NM).astype(float)
    s1, s2, nn = S1[BIX].sum(1), S2[BIX].sum(1), N[BIX].sum(1)
    return (s1 / nn) / np.sqrt((s2 - s1 * s1 / nn) / (nn - 1))


def within_month_auc(p):
    v = []
    for ix in grp.values():
        y = ev['bad'].to_numpy()[ix]
        if 0 < y.sum() < len(y):
            v.append(roc_auc_score(y, p[ix]))
    return float(np.mean(v))


jit = np.random.default_rng(11).random(n) * 1e-6
QS9 = [i / 10 for i in range(1, 10)]
rows, BS = [], {}
for m in MODELS:
    p = ev[f'p|{m}'].to_numpy()
    sc = {'V1 기대수익 순': (1 - p) * ev['r'].to_numpy() - p * ev['lgd'].to_numpy() - TB + jit,
          'V2 부도확률 낮은 순': -p + jit,
          'V4 위험조정 순': evaluate.risk_adjusted_score(p, ev['r'], ev['lgd'], TB) + jit}
    for nm, s in sc.items():
        for q in QS9:
            a = by_month_top(s, q)
            rows.append({'모델': m, '전략': nm, '승인비율': q, '건별 Sharpe': evaluate.loan_sharpe(X, a)})
            if nm == 'V2 부도확률 낮은 순' and q == 0.4:
                BS[m] = boot(a)
    log(f'{m}: 그달 안 AUC {within_month_auc(p):.4f}, {(time.time() - t0) / 60:.1f}분')
G = pd.DataFrame(rows)
s_all = evaluate.loan_sharpe(X, np.ones(n, bool))

summ = []
for m in MODELS:
    g = G[G['모델'] == m]
    best = g.loc[g['건별 Sharpe'].idxmax()]
    v2 = g[g['전략'] == 'V2 부도확률 낮은 순'].set_index('승인비율')['건별 Sharpe']
    d40 = BS[m] - BS['LightGBM 기본']
    lo, hi = np.quantile(d40, [0.025, 0.975]) if m != 'LightGBM 기본' else (np.nan, np.nan)
    a = pd.DataFrame(AUC)
    summ.append({'모델': m, 'AUC(연도 평균)': a[a['모델'] == m]['AUC'].mean(), '그달 안 AUC': within_month_auc(ev[f'p|{m}'].to_numpy()),
                 'V2 40% Sharpe': v2[0.4], 'V2 40% - LightGBM 95% 하한': lo, 'V2 40% - LightGBM 95% 상한': hi,
                 'V2 최고 승인비율': v2.idxmax(), 'V2 최고 Sharpe': v2.max(),
                 '28개 중 1위': f"{best['전략']} {best['승인비율']:.0%}", '1위 Sharpe': best['건별 Sharpe']})
S = pd.DataFrame(summ)
log(f'\n전부 승인 건별 Sharpe {s_all:.4f}')
pd.set_option('display.width', 250)
log('\n== 모델별 요약\n' + S.round(4).to_string(index=False))
log('\n== V2 부도확률 낮은 순: 승인비율 × 모델\n' + G[G['전략'] == 'V2 부도확률 낮은 순'].pivot(index='승인비율', columns='모델', values='건별 Sharpe')[MODELS].round(4).to_string())
log('\n== 모델별 전략 최고값\n' + G.groupby(['모델', '전략'])['건별 Sharpe'].max().unstack().loc[MODELS].round(4).to_string())
log(f'\n총 {(time.time() - t0) / 60:.1f}분')
G.round(5).to_csv(RESULTS / '48_model_grid.csv', index=False)
S.round(5).to_csv(RESULTS / '48_model_summary.csv', index=False)
pd.DataFrame(AUC).round(5).to_csv(RESULTS / '48_model_auc.csv', index=False)
(RESULTS / '48_model_check.log').write_text('\n'.join(LOG))
