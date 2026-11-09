"""부도확률 모델. 튜닝보다 재현성과 해석을 우선해 파라미터를 고정한다."""
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.6, min_child_samples=200, reg_lambda=5, cat_smooth=50, n_jobs=-1, verbose=-1)


def fit_lgbm_pd(X, y, seed=0, valid_frac=0.15):
    """학습 표본의 일부로 조기 종료한다(평가 연도는 쓰지 않는다)."""
    u = np.random.default_rng(seed).random(len(X)) < valid_frac
    m = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
    m.fit(X[~u], y[~u], eval_set=[(X[u], y[u])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


LOGIT_COLS = ['loan_amnt', 'r', 'term_m', 'fico_range_low', 'dti', 'annual_inc', 'open_acc', 'total_acc',
              'inq_last_6mths', 'revol_util']


def fit_logit_pd(d, y):
    """해석용 기준 모델: 수업 실습 변수 + 금리·만기·리볼빙 사용률."""
    X = d[LOGIT_COLS].astype(float)
    med = X.median()
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    m.fit(X.fillna(med), y)
    m.medians_ = med
    return m


def predict_logit(m, d):
    return m.predict_proba(d[LOGIT_COLS].astype(float).fillna(m.medians_))[:, 1]
