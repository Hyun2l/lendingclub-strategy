"""부도확률 보정. 시점 학습 표본은 결과가 일찍 확정된 대출 위주라 확률 수준이 틀린다.
미래 정보를 쓰지 않고 수준을 바로잡는 세 방법을 제공한다.

1. 단순 보정(isotonic): 학습 표본의 교차 예측값과 실제 결과로 맞춘다. 표본 편향은 그대로 남는다.
2. 만기 도달 보정: 그 시점에 이미 만기까지 지켜본 대출(편향 없음)로만 맞춘다. 대신 오래된 대출이다.
3. 생존분석 이동: 최근 발행분까지 진행 중 대출을 포함해 누적 부도율을 추정하고, 예측 평균이 그 값에 맞도록 확률 전체를 옮긴다.
"""
import numpy as np
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression


def oof_predict(fit, X, y, folds=3, seed=0):
    """교차 예측: 각 행을 그 행을 보지 않은 모델로 예측한다."""
    k = np.random.default_rng(seed).integers(0, folds, len(y))
    out = np.zeros(len(y))
    for f in range(folds):
        m = fit(X[k != f], y[k != f])
        out[k == f] = m.predict_proba(X[k == f])[:, 1]
    return out


def isotonic(p_fit, y_fit):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4)
    return iso.fit(p_fit, y_fit)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shift_to_mean(p, target):
    """로짓에 같은 상수를 더해 평균 예측이 target이 되게 하는 상수."""
    z = _logit(p)
    f = lambda c: (1 / (1 + np.exp(-(z + c)))).mean() - target
    return brentq(f, -10, 10)


def apply_shift(p, c):
    return 1 / (1 + np.exp(-(_logit(p) + c)))


def cumulative_default(time, event, horizon):
    """경쟁위험(부도 vs 상환) 누적 부도율(Aalen-Johansen). event: 1 부도, 2 상환, 0 관측 중단."""
    time, event = np.asarray(time), np.asarray(event)
    S, cif = 1.0, 0.0
    for t in range(1, horizon + 1):
        at_risk = (time >= t).sum()
        if at_risk == 0:
            break
        d1 = ((time == t) & (event == 1)).sum()
        d2 = ((time == t) & (event == 2)).sum()
        cif += S * d1 / at_risk
        S *= 1 - (d1 + d2) / at_risk
    return cif
