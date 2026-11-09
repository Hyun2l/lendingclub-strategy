"""21단계(탐색): 피처 엔지니어링 x 모델 8종을 같은 시점 학습 조건에서 비교한다.

연도 Y(2013~2016)마다 Y년 1월 1일까지 확정된 대출로 학습하고, Y년 발행 36개월 종결 대출로 평가한다.
확률적 모델은 시드 0~2. 지표: AUC, Brier, ECE, 선별 가치(기대수익 순위, 무작위 대비), 등급 안 선별 효과(상위 50%).
이미 본 2013~2016년 구간이므로 결과는 참고용이며, test 프로토콜(v1)의 모델은 바꾸지 않는다.
"""
import time
import warnings

import _path  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lc import data, asof, models, evaluate, returns, features, woe
from lc.config import RESULTS

warnings.filterwarnings('ignore')
FEE = 0.01
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2]
t0 = time.time()

raw = data.load()
d = data.parse(raw)
d = d[~d['policy_fail']]
feats = data.feature_columns(d)
X0 = data.model_frame(d, feats)
num = [c for c in feats if pd.api.types.is_numeric_dtype(X0[c])]
cats = [c for c in feats if c not in num]

# ---------- 피처 엔지니어링 ----------
E = features.engineered(d, raw.loc[d.index, 'earliest_cr_line']).drop(columns=['noise', 'fico_avg'])
inc = d['annual_inc'].astype(float).where(d['annual_inc'] > 0)
E['revol_bal_inc'] = (d['revol_bal'].astype(float) / inc).clip(upper=5)
E['bal_ex_mort_inc'] = (d['total_bal_ex_mort'].astype(float) / inc).clip(upper=10)
E['open_ratio'] = d['open_acc'].astype(float) / d['total_acc'].astype(float).where(d['total_acc'] > 0)
E['bc_actv_ratio'] = d['num_actv_bc_tl'].astype(float) / d['num_bc_tl'].astype(float).where(d['num_bc_tl'] > 0)
for c in ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_recent_inq', 'mths_since_last_major_derog', 'il_util', 'emp_length']:
    E[f'na_{c}'] = d[c].isna().astype('float32')
for c in ['annual_inc', 'revol_bal', 'tot_cur_bal', 'total_rev_hi_lim', 'avg_cur_bal', 'bc_open_to_buy']:
    E[f'log_{c}'] = np.log1p(d[c].astype(float).clip(lower=0))
E = E.astype('float32')
FE = list(E.columns)
print('피처 엔지니어링', len(FE), '개:', FE, flush=True)

LOGIT_CATS = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'emp_length']


def logit_pipe(num_cols):
    pre = ColumnTransformer([('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num_cols),
                             ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.005), LOGIT_CATS)])
    return make_pipeline(pre, LogisticRegression(max_iter=3000, C=1.0))


def logit_frame(idx, with_fe):
    f = X0.loc[idx, num].copy()
    for c in LOGIT_CATS:
        f[c] = d.loc[idx, c].astype(str)
    if with_fe:
        f = f.join(E.loc[idx])
    return f


def ece(p, y, bins=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), bins, labels=False)
    g = pd.DataFrame({'p': p, 'y': y, 'q': q.to_numpy()}).groupby('q')
    return float((g.size() / len(p) * (g['p'].mean() - g['y'].mean()).abs()).sum())


def brinson_sel(e, a):
    w = e['funded_amnt'].to_numpy()
    alloc = sel = 0.0
    bench = returns.pooled_irr(e, fee=FEE)
    for _, ix in e.groupby('grade', observed=True).indices.items():
        wb = w[ix].sum() / w.sum()
        sa = ix[a[ix]]
        ws = w[sa].sum() / w[a].sum() if len(sa) else 0.0
        rb = returns.pooled_irr(e.iloc[ix], fee=FEE)
        rs = returns.pooled_irr(e.iloc[sa], fee=FEE) if len(sa) >= 30 else rb
        alloc += (ws - wb) * (rb - bench)
        sel += wb * (rs - rb)
    return alloc, sel


def curve(e, s):
    return np.array([(lambda c: (c['irr'] - c['tb']).mean())(evaluate.cohort_irr(e, s >= np.quantile(s, 1 - q), fee=FEE)) for q in QS])


LGB_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 63, 255) for mc in (50, 500)]


def fit_lgb(Xt, yt, seed, extra=None):
    p = dict(models.LGB_PARAMS)
    p.update(extra or {})
    u = np.random.default_rng(seed).random(len(Xt)) < 0.15
    m = lgb.LGBMClassifier(random_state=seed, **p)
    m.fit(Xt[~u], yt[~u], eval_set=[(Xt[u], yt[u])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


rows = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri = tr.index
    ytr = d.loc[tri, 'bad'].to_numpy()
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    yev = ev['bad'].to_numpy()
    lgd = asof.loss_given_default(tr)
    rnd = np.mean([curve(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=0)
    preds = {}

    # 로지스틱 3종(결정적이라 시드 1개)
    m = logit_pipe(num).fit(logit_frame(tri, False), ytr)
    preds[('로지스틱 원본', 0)] = m.predict_proba(logit_frame(evi, False))[:, 1]
    m = logit_pipe(num + FE).fit(logit_frame(tri, True), ytr)
    preds[('로지스틱 +피처', 0)] = m.predict_proba(logit_frame(evi, True))[:, 1]
    W = woe.WOE(n_bins=10).fit(d.loc[tri, feats].join(E.loc[tri]), ytr, feats + FE)
    keep = [c for c, v in W.iv_table().items() if v >= 0.02]
    Wt, We = W.transform(d.loc[tri, feats].join(E.loc[tri]), keep), W.transform(d.loc[evi, feats].join(E.loc[evi]), keep)
    m = LogisticRegression(max_iter=3000, C=1.0).fit(Wt, ytr)
    preds[('로지스틱 WOE', 0)] = m.predict_proba(We)[:, 1]
    print(Y, '로지스틱 완료', f'WOE 변수 {len(keep)}개', f'{(time.time() - t0) / 60:.1f}분', flush=True)

    # LightGBM 튜닝: 학습 표본 안에서 무작위 20%를 검증으로 써서 설정 선택(AUC)
    u = np.random.default_rng(Y).random(len(tri)) < 0.2
    best, best_auc = None, -1
    for cfg in LGB_GRID:
        mm = fit_lgb(X0.loc[tri[~u]], ytr[~u], 0, cfg)
        a = roc_auc_score(ytr[u], mm.predict_proba(X0.loc[tri[u]])[:, 1])
        if a > best_auc:
            best, best_auc = cfg, a
    print(Y, '튜닝 선택', best, round(best_auc, 4), flush=True)

    XtF, XeF = X0.loc[tri].join(E.loc[tri]), X0.loc[evi].join(E.loc[evi])
    Xt_codes = X0.loc[tri].copy()
    Xe_codes = X0.loc[evi].copy()
    for c in cats:
        Xt_codes[c] = Xt_codes[c].cat.codes.astype('float32')
        Xe_codes[c] = Xe_codes[c].cat.codes.astype('float32')
    # HistGB·RF는 학습 기간에 값이 없거나 하나뿐인 열(아직 생기지 않은 변수)을 처리하지 못해 뺀다
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    for s in SEEDS:
        preds[('LightGBM 기본', s)] = fit_lgb(X0.loc[tri], ytr, s).predict_proba(X0.loc[evi])[:, 1]
        preds[('LightGBM +피처', s)] = fit_lgb(XtF, ytr, s).predict_proba(XeF)[:, 1]
        preds[('LightGBM 튜닝', s)] = fit_lgb(X0.loc[tri], ytr, s, best).predict_proba(X0.loc[evi])[:, 1]
        hg = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, categorical_features='from_dtype',
                                            random_state=s).fit(X0.loc[tri, usable], ytr)
        preds[('HistGradientBoosting', s)] = hg.predict_proba(X0.loc[evi, usable])[:, 1]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features='sqrt', max_samples=0.3, n_jobs=-1,
                                    random_state=s).fit(Xt_codes[usable], ytr)
        preds[('RandomForest', s)] = rf.predict_proba(Xe_codes[usable])[:, 1]
        print(Y, '시드', s, '완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)

    for (nm, s), p in preds.items():
        score = (1 - p) * ev['r'].to_numpy() - p * lgd - ev['tb'].to_numpy()
        cv = curve(ev, score)
        a50 = score >= np.quantile(score, 0.5)
        al, se = brinson_sel(ev, a50)
        rows.append({'연도': Y, '모델': nm, '시드': s, 'AUC': roc_auc_score(yev, p), 'Brier': brier_score_loss(yev, p), 'ECE': ece(p, yev),
                     '선별 가치': float((cv - rnd).mean()), '상위50% 초과IRR': float(cv[QS.index(0.5)]), '등급 배분': al, '등급 안 선별': se})
    pd.DataFrame(rows).round(5).to_csv(RESULTS / '21_fe_models_rows.csv', index=False)

R = pd.DataFrame(rows)
by_year = R.groupby(['모델', '연도'])[['AUC', '선별 가치', '등급 안 선별']].mean()
S = R.groupby('모델').agg(AUC=('AUC', 'mean'), Brier=('Brier', 'mean'), ECE=('ECE', 'mean'), 선별가치=('선별 가치', 'mean'),
                         등급안선별=('등급 안 선별', 'mean'), 등급배분=('등급 배분', 'mean'))
seed_rng = R.groupby(['모델', '연도'])['선별 가치'].agg(lambda x: x.max() - x.min()).groupby('모델').max()
S['선별가치 시드폭(최대)'] = seed_rng
base = by_year.xs('LightGBM 기본', level='모델')['선별 가치']
S['기본보다 나은 연도'] = [int((by_year.xs(m, level='모델')['선별 가치'] > base).sum()) for m in S.index]
S = S.sort_values('선별가치', ascending=False)
S.round(5).to_csv(RESULTS / '21_fe_models_summary.csv')
by_year.round(5).to_csv(RESULTS / '21_fe_models_by_year.csv')
pd.set_option('display.width', 220)
print(S.round(4).to_string())
print(by_year['선별 가치'].unstack().round(4).to_string())
print(f'총 {(time.time() - t0) / 60:.1f}분')
