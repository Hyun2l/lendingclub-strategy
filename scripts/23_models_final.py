"""23단계(탐색): 결측 방침(22·22b단계에서 확정)을 적용한 최종 모델 비교. Ridge, Lasso, Elastic Net과 중복 변수 처리 포함.

공통: 시점 학습 2013~2016, 평가는 연도별 36개월 종결 대출, 투자자 수수료 1%.
결측 방침: 트리 모델은 결측 그대로. 선형 모델은 학습 가용률 50% 미만 변수 제외, 1~99% 분위로 자르기, 중앙값 대치,
범주형은 원-핫(근속 연수 결측은 별도 범주).

선형 모델(규제 강도 C는 학습 데이터 안의 무작위 20% 검증에서 로그 손실 최소로 선택, 평가 연도는 쓰지 않음)
- L2 기준: C=1 고정(sklearn 기본값)
- 중복 제거 + L2: 순위상관 0.9 이상 쌍에서 IV가 낮은 쪽을 뺌
- Ridge(L2, C 선택) / Lasso(L1, C 선택) / Elastic Net(l1_ratio 0.5, C 선택)
- L2 + 피처 엔지니어링 / WOE 스코어카드
트리 모델(시드 0~2): LightGBM 기본, LightGBM + 피처, LightGBM 튜닝, HistGradientBoosting, RandomForest

판정 기준(미리 정함)
1) 중복 변수·규제: 'L2 기준'보다 선별 가치가 0.1%p 넘게 높거나 AUC가 0.002 넘게 높은 연도가 3개 이상이면 채택, 아니면 기준 유지.
2) 모델 비교: 'LightGBM 기본'보다 위 기준으로 3개 연도 이상 나은 모델이 있으면 '도전자'로 기록한다.
   test 프로토콜(v1)의 모델은 바꾸지 않는다(이미 본 구간에서의 탐색이므로).
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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lc import data, asof, models, evaluate, returns, features, woe
from lc.config import RESULTS

warnings.filterwarnings('ignore')
FEE = 0.01
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2]
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
t0 = time.time()

raw = data.load()
d = data.parse(raw)
d = d[~d['policy_fail']]
feats = data.feature_columns(d)
X0 = data.model_frame(d, feats)
cats_all = [c for c in feats if not pd.api.types.is_numeric_dtype(X0[c])]
LOGIT_CATS = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'emp_length']

# 피처 엔지니어링(21단계와 같음)
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


class Linear:
    """선형 모델 전처리(결측 방침) + 로지스틱. 기준값은 학습 데이터에서만."""

    def __init__(self, penalty='l2', C=1.0, l1_ratio=None):
        self.penalty, self.C, self.l1_ratio = penalty, C, l1_ratio

    def _prep(self, X, fit=False):
        X = X.copy()
        if fit:
            self.cats = [c for c in LOGIT_CATS if c in X.columns]
            num = [c for c in X.columns if c not in self.cats and pd.api.types.is_numeric_dtype(X[c])]
            self.num = [c for c in num if X[c].notna().mean() >= 0.5 and X[c].nunique(dropna=True) > 1]   # 가용성 규칙
            self.lo, self.hi = X[self.num].quantile(0.01), X[self.num].quantile(0.99)
        X[self.num] = X[self.num].astype(float).clip(self.lo, self.hi, axis=1)
        for c in self.cats:
            X[c] = X[c].astype(str)
        return X[self.num + self.cats]

    def fit(self, X, y):
        Xp = self._prep(X, fit=True)
        pre = ColumnTransformer([('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), self.num),
                                 ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.005), self.cats)])
        solver = 'lbfgs' if self.penalty == 'l2' else 'saga'
        kw = dict(C=self.C, solver=solver, max_iter=3000 if solver == 'lbfgs' else 300, tol=1e-4 if solver == 'lbfgs' else 1e-3)
        if self.penalty == 'l1':
            kw['l1_ratio'] = 1.0
        elif self.penalty == 'elasticnet':
            kw['l1_ratio'] = self.l1_ratio
        self.m = make_pipeline(pre, LogisticRegression(**kw)).fit(Xp, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(self._prep(X))[:, 1]

    def nonzero(self):
        return int((np.abs(self.m[-1].coef_) > 1e-8).sum()), int(self.m[-1].coef_.size)


def choose_C(X, y, penalty, l1_ratio=None, seed=0):
    u = np.random.default_rng(seed).random(len(X)) < 0.2
    best, best_ll = None, np.inf
    for C in CS:
        p = Linear(penalty, C, l1_ratio).fit(X[~u], y[~u]).predict(X[u])
        ll = log_loss(y[u], np.clip(p, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best, best_ll = C, ll
    return best


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


def fit_lgb(Xt, yt, seed, extra=None):
    p = dict(models.LGB_PARAMS)
    p.update(extra or {})
    u = np.random.default_rng(seed).random(len(Xt)) < 0.15
    m = lgb.LGBMClassifier(random_state=seed, **p)
    m.fit(Xt[~u], yt[~u], eval_set=[(Xt[u], yt[u])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


LGB_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 63, 255) for mc in (50, 500)]
rows, chosen = [], []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    ev = returns.attach_treasury(d.loc[evi]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    rnd = np.mean([curve(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=0)
    preds = {}
    Xt, Xe = X0.loc[tri, usable], X0.loc[evi, usable]

    # ---- 선형 모델 ----
    preds[('L2 기준(C=1)', 0)] = Linear('l2', 1.0).fit(Xt, ytr).predict(Xe)
    # 중복 제거: 학습 데이터에서 순위상관 0.9 이상 쌍 중 IV 낮은 쪽 제외
    numc = [c for c in usable if c not in cats_all and Xt[c].notna().mean() >= 0.5]
    W0 = woe.WOE(n_bins=10).fit(d.loc[tri, numc], ytr, numc)
    ivs = W0.iv_table()
    corr = Xt[numc].sample(min(100000, len(Xt)), random_state=0).rank().corr().abs()
    drop = set()
    for i, a in enumerate(numc):
        for b in numc[i + 1:]:
            if corr.loc[a, b] >= 0.9 and a not in drop and b not in drop:
                drop.add(a if ivs.get(a, 0) < ivs.get(b, 0) else b)
    keepc = [c for c in usable if c not in drop]
    preds[('중복 제거 + L2', 0)] = Linear('l2', 1.0).fit(Xt[keepc], ytr).predict(Xe[keepc])
    for name, pen, l1r in [('Ridge(C 선택)', 'l2', None), ('Lasso(C 선택)', 'l1', None), ('Elastic Net(C 선택)', 'elasticnet', 0.5)]:
        C = choose_C(Xt, ytr, pen, l1r)
        m = Linear(pen, C, l1r).fit(Xt, ytr)
        preds[(name, 0)] = m.predict(Xe)
        nz, tot = m.nonzero()
        chosen.append({'연도': Y, '모델': name, 'C': C, '0이 아닌 계수': nz, '전체 계수': tot})
    chosen.append({'연도': Y, '모델': '중복 제거 + L2', 'C': 1.0, '0이 아닌 계수': len(drop), '전체 계수': len(usable)})
    XtF, XeF = Xt.join(E.loc[tri]), Xe.join(E.loc[evi])
    preds[('L2 + 피처', 0)] = Linear('l2', 1.0).fit(XtF, ytr).predict(XeF)
    W = woe.WOE(n_bins=10).fit(d.loc[tri, usable], ytr, usable)
    keep = [c for c, v in W.iv_table().items() if v >= 0.02]
    preds[('WOE 스코어카드', 0)] = LogisticRegression(max_iter=3000).fit(W.transform(d.loc[tri, usable], keep), ytr) \
        .predict_proba(W.transform(d.loc[evi, usable], keep))[:, 1]
    print(Y, '선형 완료', f'{(time.time() - t0) / 60:.1f}분', chosen[-4:], flush=True)

    # ---- 트리 모델 ----
    u = np.random.default_rng(Y).random(len(tri)) < 0.2
    best, best_auc = None, -1
    for cfg in LGB_GRID:
        mm = fit_lgb(Xt[~u], ytr[~u], 0, cfg)
        a = roc_auc_score(ytr[u], mm.predict_proba(Xt[u])[:, 1])
        if a > best_auc:
            best, best_auc = cfg, a
    chosen.append({'연도': Y, '모델': 'LightGBM 튜닝', 'C': str(best)})
    Xt_codes, Xe_codes = Xt.copy(), Xe.copy()
    for c in [c for c in cats_all if c in usable]:
        Xt_codes[c], Xe_codes[c] = Xt_codes[c].cat.codes.astype('float32'), Xe_codes[c].cat.codes.astype('float32')
    for s in SEEDS:
        preds[('LightGBM 기본', s)] = fit_lgb(Xt, ytr, s).predict_proba(Xe)[:, 1]
        preds[('LightGBM + 피처', s)] = fit_lgb(XtF, ytr, s).predict_proba(XeF)[:, 1]
        preds[('LightGBM 튜닝', s)] = fit_lgb(Xt, ytr, s, best).predict_proba(Xe)[:, 1]
        preds[('HistGradientBoosting', s)] = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True,
                                                                           categorical_features='from_dtype', random_state=s).fit(Xt, ytr).predict_proba(Xe)[:, 1]
        preds[('RandomForest', s)] = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features='sqrt', max_samples=0.3,
                                                            n_jobs=-1, random_state=s).fit(Xt_codes, ytr).predict_proba(Xe_codes)[:, 1]
        print(Y, '트리 시드', s, f'{(time.time() - t0) / 60:.1f}분', flush=True)

    for (nm, s), p in preds.items():
        score = (1 - p) * ev['r'].to_numpy() - p * lgd - ev['tb'].to_numpy()
        cv = curve(ev, score)
        rows.append({'연도': Y, '모델': nm, '시드': s, 'AUC': roc_auc_score(ev['bad'], p), 'Brier': brier_score_loss(ev['bad'], p),
                     '선별 가치': float((cv - rnd).mean()), '등급 안 선별': within(ev, score >= np.quantile(score, 0.5))})
    pd.DataFrame(rows).round(5).to_csv(RESULTS / '23_models_rows.csv', index=False)
    pd.DataFrame(chosen).to_csv(RESULTS / '23_models_chosen.csv', index=False)

R = pd.DataFrame(rows)
Yr = R.groupby(['모델', '연도'])[['AUC', '선별 가치', '등급 안 선별', 'Brier']].mean()
noise = R[R['모델'] == 'LightGBM 기본'].groupby('연도')['선별 가치'].agg(lambda x: x.max() - x.min())
print('LightGBM 기본의 연도별 시드 폭(선별 가치)', noise.round(4).to_dict())


def compare(base_name):
    base = Yr.xs(base_name, level='모델')
    out = []
    for m in R['모델'].unique():
        t = Yr.xs(m, level='모델')
        better = ((t['선별 가치'] - base['선별 가치'] > 0.001) | (t['AUC'] - base['AUC'] > 0.002)).sum()
        worse = ((base['선별 가치'] - t['선별 가치'] > 0.001) | (base['AUC'] - t['AUC'] > 0.002)).sum()
        out.append({'모델': m, 'AUC': t['AUC'].mean(), '선별 가치': t['선별 가치'].mean(), '등급 안 선별': t['등급 안 선별'].mean(),
                    'Brier': t['Brier'].mean(), f'{base_name}보다 나은 연도': int(better), '나쁜 연도': int(worse)})
    return pd.DataFrame(out).sort_values('선별 가치', ascending=False)


S_lin = compare('L2 기준(C=1)')
S_all = compare('LightGBM 기본')
S_lin.round(5).to_csv(RESULTS / '23_models_vs_linear_baseline.csv', index=False)
S_all.round(5).to_csv(RESULTS / '23_models_vs_lgbm.csv', index=False)
Yr.round(5).to_csv(RESULTS / '23_models_by_year.csv')
pd.set_option('display.width', 230)
print(S_all.round(4).to_string(index=False))
print(S_lin[S_lin['모델'].isin(['L2 기준(C=1)', '중복 제거 + L2', 'Ridge(C 선택)', 'Lasso(C 선택)', 'Elastic Net(C 선택)', 'L2 + 피처', 'WOE 스코어카드'])].round(4).to_string(index=False))
print(Yr['선별 가치'].unstack().round(4).to_string())
print(pd.DataFrame(chosen).to_string(index=False))
print(f'총 {(time.time() - t0) / 60:.1f}분')
