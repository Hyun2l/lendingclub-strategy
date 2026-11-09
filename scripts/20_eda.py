"""20단계: 보고서용 EDA. 그림 5장과 표 3개를 results/eda/ 에 만든다.
표본: 2007~2016년 발행 종결 대출(결과가 대부분 확정된 구간). 기술 통계라 모델 선택에는 쓰지 않는다."""
import _path  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lc import data, woe
from lc.config import RESULTS

plt.rcParams.update({'font.family': 'Nanum Gothic', 'axes.unicode_minus': False, 'font.size': 9})
OUT = RESULTS / 'eda'
OUT.mkdir(parents=True, exist_ok=True)
d = data.parse(data.load())
r = data.resolved_sample(d)
r = r[r['issue_year'] <= 2016]
feats = data.feature_columns(d)
num = [c for c in feats if pd.api.types.is_numeric_dtype(r[c]) and r[c].dtype != bool]
print('표본', len(r), '| 특성', len(feats), '(숫자', len(num), ')')

# 그림 1: 연도별 건수와 부도율
g = d[d['issue_year'].between(2007, 2020)].groupby('issue_year')
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(g.size().index, g.size().values / 1000, color='0.8', label='발행 건수(천 건)')
ax2 = ax.twinx()
rr = data.resolved_sample(d).groupby('issue_year')['bad'].mean()
ax2.plot(rr.index, rr.values * 100, 'k-o', ms=3, label='종결 대출 부도율(%)')
ax.set_xlabel('발행연도'); ax.set_ylabel('천 건'); ax2.set_ylabel('부도율 %')
ax.set_title('발행 건수와 종결 대출 부도율(2017년 이후는 종결 편향)')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.88), frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'fig1_year.png', dpi=160); plt.close(fig)

# 그림 2: 세부등급별 부도율과 평균 금리
s = r.groupby('sub_grade', observed=True).agg(부도율=('bad', 'mean'), 금리=('r', 'mean'), 건수=('bad', 'size'))
s = s[s['건수'] >= 200]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(range(len(s)), s['부도율'] * 100, 'k-', label='부도율(%)')
ax.plot(range(len(s)), s['금리'] * 100, '--', color='0.5', label='평균 금리(%)')
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=90, fontsize=6)
ax.set_title('세부등급별 부도율과 금리: 금리는 위험을 따라 오르지만 기울기가 다르다'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'fig2_subgrade.png', dpi=160); plt.close(fig)
s.round(4).to_csv(OUT / 'T_subgrade.csv')

# IV 표(전체 특성, 학습 표본 기준)
W = woe.WOE(n_bins=10).fit(r, r['bad'].to_numpy(), feats)
iv = W.iv_table().rename('IV').to_frame()
iv['강도'] = iv['IV'].map(woe.iv_strength)
iv['결측률'] = r[iv.index].isna().mean()
iv.round(4).to_csv(OUT / 'T_iv.csv')
print(iv.head(25).round(3).to_string())
print(iv['강도'].value_counts().to_string())

# 그림 3: IV 상위 숫자 특성 9개의 10분위별 부도율
top = [c for c in iv.index if c in num][:9]
fig, axes = plt.subplots(3, 3, figsize=(9, 7))
for ax, c in zip(axes.ravel(), top):
    x = r[c]
    q = pd.qcut(x.rank(method='first'), 10, labels=False)
    t = r.groupby(q)['bad'].mean() * 100
    ax.plot(t.index + 1, t.values, 'k-o', ms=3)
    if x.isna().any():
        ax.axhline(r.loc[x.isna(), 'bad'].mean() * 100, color='0.5', ls='--', lw=1)
    ax.set_title(f'{c} (IV {iv.loc[c, "IV"]:.2f})', fontsize=8); ax.set_xlabel('10분위', fontsize=7)
fig.suptitle('IV 상위 숫자 특성: 10분위별 부도율(%)  점선 = 결측일 때', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'fig3_top_numeric.png', dpi=160); plt.close(fig)

# 그림 4: 범주형
cats = ['purpose', 'home_ownership', 'verification_status', 'emp_length']
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, c in zip(axes.ravel(), cats):
    t = r.groupby(r[c].astype('string').fillna('결측'))['bad'].agg(['mean', 'size'])
    t = t[t['size'] >= 500].sort_values('mean')
    ax.barh(t.index, t['mean'] * 100, color='0.6')
    ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7); ax.set_xlabel('부도율 %', fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'fig4_categorical.png', dpi=160); plt.close(fig)

# 그림 5와 표: 상관(스피어만) 상위 20개 숫자 특성, 0.9 이상 쌍
top20 = [c for c in iv.index if c in num][:20]
C = r[top20].rank().corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(C.values, cmap='Greys', vmin=-1, vmax=1)
ax.set_xticks(range(len(top20))); ax.set_xticklabels(top20, rotation=90, fontsize=6)
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20, fontsize=6)
fig.colorbar(im, fraction=0.04); ax.set_title('IV 상위 20개 숫자 특성의 순위상관')
fig.tight_layout(); fig.savefig(OUT / 'fig5_corr.png', dpi=160); plt.close(fig)
Call = r[num].sample(200000, random_state=0).rank().corr()
pairs = Call.where(np.triu(np.ones(Call.shape, bool), 1)).stack()
hi = pairs[pairs.abs() >= 0.9].sort_values(key=abs, ascending=False).rename('순위상관').to_frame()
hi.round(3).to_csv(OUT / 'T_high_corr_pairs.csv')
print('순위상관 0.9 이상 쌍', len(hi)); print(hi.head(15).round(3).to_string())
