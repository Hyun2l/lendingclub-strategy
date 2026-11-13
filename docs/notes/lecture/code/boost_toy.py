"""보고서 5장 예시: 대출 6건으로 그래디언트 부스팅 두 번 돌리기(로그 손실, 뉴턴 잎 값, 학습률 0.5)."""
import numpy as np
fico_low = np.array([1, 1, 1, 0, 0, 0])      # FICO 690 미만이면 1
acc_many = np.array([1, 0, 1, 1, 0, 0])      # 최근 계좌 개설 많으면 1
y = np.array([1, 0, 1, 0, 0, 0])             # 부도 1
lr = 0.5
z = np.full(6, np.log(y.mean() / (1 - y.mean())))
print('출발 z', z[0].round(4), 'p', (1 / (1 + np.exp(-z[0]))).round(4))
for t, x in enumerate([fico_low, acc_many], 1):
    p = 1 / (1 + np.exp(-z))
    g = y - p                     # 잔차(로그 손실의 음의 기울기)
    h = p * (1 - p)
    for v in (0, 1):
        m = x == v
        leaf = g[m].sum() / h[m].sum()
        print(f'나무{t} 칸 x={v}: 잔차합 {g[m].sum():.4f}, 곡률합 {h[m].sum():.4f}, 잎 값 {leaf:.4f}, 더하는 값 {lr*leaf:.4f}')
        z[m] += lr * leaf
    p = 1 / (1 + np.exp(-z))
    print(f'나무{t} 뒤 p', p.round(3), '로그손실', round(float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()), 4))
