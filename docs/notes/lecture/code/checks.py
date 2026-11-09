"""강의노트 예제 숫자 검산. 모든 숫자는 예시(illustrative)."""
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

# 1) 원리금 균등 상환액: P=10000, 연 12%, 36개월
P, r, T = 10000, 0.12, 36
i = r / 12
inst = P * i / (1 - (1 + i) ** -T)
print('1 월 상환액', round(inst, 2))

def irr(cf):
    f = lambda m: (cf / (1 + m) ** np.arange(len(cf))).sum()
    m = brentq(f, -0.95, 1.0)
    return m, (1 + m) ** 12 - 1

# 2) 정상 상환 IRR = 약정 금리(월 1%)
cf = np.r_[-P, np.full(T, inst)]
print('2 정상 상환 월 IRR, 연 IRR', [round(x, 5) for x in irr(cf)])
# 3) 10개월 내고 부도, 15개월째 회수 500 (상각 지연 5개월 표현)
cf = np.zeros(16); cf[0] = -P; cf[1:11] = inst; cf[15] = 500
tot = 10 * inst + 500
print('3 받은 총액', round(tot, 2), '손실률 1-받은/대출', round(1 - tot / P, 4), '연 IRR', round(irr(cf)[1], 4))
# 단순 연환산(우승 논문식)
print('3 단순 연환산', round((tot / P) ** (12 / 36) - 1, 4))
# 4) 포트폴리오: 정상 9건 + 부도 1건(위 대출), 수수료 1%
H = 37
cfs = np.zeros(H); cfs[0] = -10 * P; cfs[1:] += 9 * inst
cfs[1:11] += inst; cfs[15] += 500
cfs[1:] *= 0.99
print('4 포트폴리오 연 IRR', round(irr(cfs)[1], 4))

# 5) 기대수익 점수: s = (1-p) r - p LGD - tb
for p_, r_ in [(0.05, 0.08), (0.25, 0.22)]:
    print('5 점수', p_, r_, round((1 - p_) * r_ - p_ * 0.6 - 0.015, 4))

# 6) AUC: 부도 2건, 정상 3건
y = np.array([1, 1, 0, 0, 0]); p = np.array([0.30, 0.10, 0.20, 0.05, 0.08])
print('6 AUC', roc_auc_score(y, p))
# Brier
print('6 Brier', round(np.mean((p - y) ** 2), 4))

# 7) Sharpe sqrt(a) 항등식: 승인 a, 초과수익 평균 m, 표준편차 s, 거절 0
m, s = 0.04, 0.20
for a in [1.0, 0.5, 0.1]:
    print('7 SR', a, round(np.sqrt(a) * m / np.sqrt(s ** 2 + (1 - a) * m ** 2), 4))

# 8) WOE/IV: 구간 2개
bad = np.array([30, 70]); good = np.array([270, 630])  # 결측 구간, 값 있음 구간
B, G = bad.sum(), good.sum()
woe = np.log((good / G) / (bad / B))
iv = ((good / G - bad / B) * woe).sum()
print('8 WOE', woe.round(4), 'IV', round(iv, 4))
bad = np.array([60, 40]); good = np.array([240, 660])
woe = np.log((good / G) / (bad / B)); iv = ((good / G - bad / B) * woe).sum()
print('8b WOE', woe.round(4), 'IV', round(iv, 4))

# 9) Brinson: 등급 2개, 비중 w, 수익 R (기준=전부 승인, 포트=모델)
wb = np.array([0.6, 0.4]); Rb = np.array([0.04, 0.07])
wp = np.array([0.7, 0.3]); Rp = np.array([0.05, 0.08])
alloc = ((wp - wb) * Rb).sum(); sel = (wb * (Rp - Rb)).sum(); inter = ((wp - wb) * (Rp - Rb)).sum()
print('9 기준', (wb * Rb).sum(), '포트', (wp * Rp).sum(), '배분', round(alloc, 4), '선별', round(sel, 4), '상호작용', round(inter, 4))

# 10) 종결 편향 장난감: 4건, 1년 뒤 관측
# A: 저금리 6%, 정상(3년 걸림)  B: 고금리 20%, 정상(3년)  C: 고금리 20%, 6개월 뒤 부도  D: 저금리 6%, 1년 안 조기상환
print('10 전체 부도율 1/4, 1년 관측 남은 대출 C,D -> 부도율 1/2')

# 11) 부트스트랩 장난감: 월별 차이 5개
dlt = np.array([0.012, 0.008, -0.002, 0.010, 0.006])
rng = np.random.default_rng(0)
bs = dlt[rng.integers(0, 5, (10000, 5))].mean(1)
print('11 평균', dlt.mean(), '95% 구간', np.quantile(bs, [0.025, 0.975]).round(4))

# 12) Holm: p = 0.01, 0.02, 0.04 (k=3)
ps = [0.01, 0.02, 0.04]
print('12 Holm', [min(1, p * (3 - j)) for j, p in enumerate(ps)])

# 13) 로지스틱: z = -2 + 0.5 x
for x in [0, 2, 4]:
    print('13 시그모이드', x, round(1 / (1 + np.exp(-(-2 + 0.5 * x))), 4))
