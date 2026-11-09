"""은행 관점 RAROC(위험조정 자본수익률).

RAROC = (금리 - 연 예상손실 - 조달비용 - 운영비) / 위험자본
- 연 예상손실 = 연 부도확률 x 손실률
- 위험자본 = 바젤 IRB '기타 소매' 자본요구량 K x 배수
  상관계수 R = 0.03 w + 0.16 (1 - w),  w = (1 - e^(-35 PD)) / (1 - e^(-35))
  K = LGD x (N[(1-R)^-0.5 G(PD) + (R/(1-R))^0.5 G(0.999)] - PD)
승인 규칙: RAROC > 요구수익률 h  <=>  경제적 이익(EVA) = 금리 - 예상손실 - 조달 - 운영비 - h x 자본 > 0
"""
import numpy as np
from scipy.stats import norm


def annual_pd(p_life, term_months):
    """만기 전체 부도확률 -> 연 부도확률(매년 같은 위험 가정)."""
    return 1 - (1 - np.clip(p_life, 1e-6, 0.999)) ** (12 / np.asarray(term_months))


def basel_retail_k(pd_annual, lgd):
    pd_ = np.clip(pd_annual, 3e-4, 0.999)  # 바젤 소매 PD 하한 0.03%
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    r = 0.03 * w + 0.16 * (1 - w)
    z = norm.ppf(pd_) / np.sqrt(1 - r) + np.sqrt(r / (1 - r)) * norm.ppf(0.999)
    return np.asarray(lgd) * (norm.cdf(z) - pd_)


def components(rate, p_life, term_months, lgd, funding, opex, capital_mult=1.0):
    pa = annual_pd(p_life, term_months)
    el = pa * lgd
    cap = basel_retail_k(pa, lgd) * capital_mult
    spread = rate - el - funding - opex
    return spread, cap


def raroc(rate, p_life, term_months, lgd, funding, opex, capital_mult=1.0):
    spread, cap = components(rate, p_life, term_months, lgd, funding, opex, capital_mult)
    return spread / cap
