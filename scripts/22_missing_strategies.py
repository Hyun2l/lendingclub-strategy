"""22단계: 결측 처리 방식 비교(모델 비교보다 먼저 결정).

시점 학습 2013~2016, 평가는 연도별 36개월 종결 대출. 처리 기준값은 학습 데이터에서만 정한다.
LightGBM(시드 0~2): native / median / median_flags / typed_tree
로지스틱: median / median_flags / typed_linear / WOE(결측을 별도 구간으로)

판정 기준(미리 정함): 같은 모델 안에서, 가장 단순한 방식(LightGBM=native, 로지스틱=median)보다
선별 가치가 0.1%p 넘게 높거나 AUC가 0.002 넘게 높은 연도가 4개 중 3개 이상인 방식이 있으면 그 방식을 택한다.
없으면 가장 단순한 방식을 택한다.
"""
import time
import warnings

import _path  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lc import data, asof, models, evaluate, returns, woe
from lc.missing import MissingPolicy
from lc.config import RESULTS

warnings.filterwarnings('ignore')
FEE = 0.01
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t0 = time.time()
d = data.parse(data.load())
d = d[~d['policy_fail']]
feats = data.feature_columns(d)
X0 = data.model_frame(d, feats)
CATS = [c for c in feats if not pd.api.types.is_numeric_dtype(X0[c])]
LOGIT_CATS = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'emp_length']


def curve(e, s):
    return np.array([(lambda c: (c['irr'] - c['tb']).mean())(evaluate.cohort_irr(e, s >= np.quantile(s, 1 - q), fee=FEE)) for q in QS])


def within(e, a):
    w = e['funded_amnt'].to_numpy()
    sel = 0.0
    for _, ix in e.groupby('grade', observed=True).indices.items():
        sa = ix[a[ix]]
        rb = returns.pooled_irr(e.iloc[ix], fee=FEE)
        rs = returns.pooled_irr(e.iloc[sa], fee=FEE) if len(sa) >= 30 else rb
        sel += w[ix].sum() / w.sum() * (rs - rb)
    return sel


def fit_lgb(Xt, yt, seed):
    u = np.random.default_rng(seed).random(len(Xt)) < 0.15
    m = lgb.LGBMClassifier(random_state=seed, **models.LGB_PARAMS)
    m.fit(Xt[~u], yt[~u], eval_set=[(Xt[u], yt[u])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def logit_fit_predict(Xt, yt, Xe):
    cats = [c for c in LOGIT_CATS if c in Xt.columns]  # 학습 기간에 한 값뿐인 범주(예: 2013년 신청 유형)는 빠져 있음
    num = [c for c in Xt.columns if c not in cats and pd.api.types.is_numeric_dtype(Xt[c])]
    Xt, Xe = Xt.copy(), Xe.copy()
    for c in cats:
        Xt[c], Xe[c] = Xt[c].astype(str), Xe[c].astype(str)
    pre = ColumnTransformer([('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
                             ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.005), cats)])
    m = make_pipeline(pre, LogisticRegression(max_iter=3000)).fit(Xt[num + cats], yt)
    return m.predict_proba(Xe[num + cats])[:, 1]


rows = []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]  # 학습 기간에 값이 있는 컬럼만
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    rnd = np.mean([curve(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=0)
    preds = {}
    Xt_raw, Xe_raw = X0.loc[tri, usable], X0.loc[evi, usable]
    for strat in ['native', 'median', 'median_flags', 'typed_tree']:
        pol = MissingPolicy(strat).fit(Xt_raw)
        Xt, Xe = pol.transform(Xt_raw), pol.transform(Xe_raw)
        for s in (0, 1, 2):
            preds[('LightGBM', strat, s)] = fit_lgb(Xt, ytr, s).predict_proba(Xe)[:, 1]
        print(Y, 'LightGBM', strat, f'{(time.time() - t0) / 60:.1f}분', flush=True)
    for strat in ['median', 'median_flags', 'typed_linear']:
        pol = MissingPolicy(strat).fit(Xt_raw)
        preds[('로지스틱', strat, 0)] = logit_fit_predict(pol.transform(Xt_raw), ytr, pol.transform(Xe_raw))
    W = woe.WOE(n_bins=10).fit(d.loc[tri, usable], ytr, usable)
    keep = [c for c, v in W.iv_table().items() if v >= 0.02]
    preds[('로지스틱', 'WOE', 0)] = LogisticRegression(max_iter=3000).fit(W.transform(d.loc[tri, usable], keep), ytr) \
        .predict_proba(W.transform(d.loc[evi, usable], keep))[:, 1]
    print(Y, '로지스틱 완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)
    for (model, strat, s), p in preds.items():
        score = (1 - p) * ev['r'].to_numpy() - p * lgd - ev['tb'].to_numpy()
        cv = curve(ev, score)
        rows.append({'연도': Y, '모델': model, '결측 처리': strat, '시드': s, 'AUC': roc_auc_score(ev['bad'], p),
                     'Brier': brier_score_loss(ev['bad'], p), '선별 가치': float((cv - rnd).mean()),
                     '등급 안 선별': within(ev, score >= np.quantile(score, 0.5))})
    pd.DataFrame(rows).round(5).to_csv(RESULTS / '22_missing_rows.csv', index=False)

R = pd.DataFrame(rows)
Y_ = R.groupby(['모델', '결측 처리', '연도'])[['AUC', '선별 가치', '등급 안 선별', 'Brier']].mean()
summ = []
for model, simplest in [('LightGBM', 'native'), ('로지스틱', 'median')]:
    base = Y_.loc[(model, simplest)]
    for strat in R.loc[R['모델'] == model, '결측 처리'].unique():
        t = Y_.loc[(model, strat)]
        better = ((t['선별 가치'] - base['선별 가치'] > 0.001) | (t['AUC'] - base['AUC'] > 0.002)).sum()
        worse = ((base['선별 가치'] - t['선별 가치'] > 0.001) | (base['AUC'] - t['AUC'] > 0.002)).sum()
        summ.append({'모델': model, '결측 처리': strat, 'AUC': t['AUC'].mean(), '선별 가치': t['선별 가치'].mean(),
                     '등급 안 선별': t['등급 안 선별'].mean(), 'Brier': t['Brier'].mean(),
                     '단순 방식보다 확실히 나은 연도': int(better), '확실히 나쁜 연도': int(worse)})
S = pd.DataFrame(summ)
S.round(5).to_csv(RESULTS / '22_missing_summary.csv', index=False)
pd.set_option('display.width', 220)
print(S.round(4).to_string(index=False))
print(Y_['AUC'].unstack().round(4).to_string())
print(Y_['선별 가치'].unstack().round(4).to_string())
for model, simplest in [('LightGBM', 'native'), ('로지스틱', 'median')]:
    cand = S[(S['모델'] == model) & (S['단순 방식보다 확실히 나은 연도'] >= 3)]
    pick = cand.sort_values('선별 가치', ascending=False)['결측 처리'].iloc[0] if len(cand) else simplest
    print(f'판정: {model} -> {pick}')
print(f'총 {(time.time() - t0) / 60:.1f}분')
