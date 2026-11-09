"""26단계(검증): 종결 편향 시뮬레이션.

질문: 2017년 보류 구간에서 '부도확률 순위 > 기대수익 순위'가 나온 것은 일찍 끝난 대출만 보이는 종결 편향 때문인가?
공식 test에도 진행 중 대출이 없으므로, test 결과를 어떻게 읽을지를 test를 보기 전에 정해 두려는 실험이다.

방법:
- 2013~2015년 발행 36개월 대출은 거의 모두 끝나 정답을 안다. 같은 대출을 관측 시점 T에 '이미 결과가 확정된 대출'만 남기도록 자른다.
  T = 발행 다음다음 해 1월 1일(발행 후 약 1~2년), 그다음 해 1월 1일(약 2~3년), 전체.
- 모델은 연도별 시점 학습 LightGBM 기본(시드 10개). 예측은 자르기와 무관하고, 평가 표본만 바뀐다.
- 각 T에서 기대수익 순위와 부도확률 순위의 선별 가치(같은 표본의 무작위 승인 대비)와 그 차이를 구한다.
  차이의 95% 구간은 발행월 부트스트랩(1만 번)으로 구한다.

판정(사전 고정):
- 전체 표본에서는 기대수익 순위가 이기고(구간 > 0), 가장 짧게 자른 표본에서 차이가 0 이하로 내려가면
  '2017년 결과는 종결 편향으로 설명된다'고 본다.
- 전체 표본에서도 차이가 0을 포함하면 C6(기대수익 순위 우위)를 약하게 고친다.
"""
import time
import warnings

import _path  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
from lc import data, asof, models, evaluate, returns
from lc.config import RESULTS

warnings.filterwarnings('ignore')
FEE, QS, B = 0.01, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10000
SEEDS = range(10)
t0 = time.time()
d = data.parse(data.load())
d = d[~d['policy_fail']]
feats = data.feature_columns(d)
X0 = data.model_frame(d, feats)


def monthly(e, s):
    return pd.concat([(lambda c: c['irr'] - c['tb'])(evaluate.cohort_irr(e, s >= np.quantile(s, 1 - q), fee=FEE)) for q in QS], axis=1).mean(axis=1)


def fit_lgb(Xt, yt, seed):
    u = np.random.default_rng(seed).random(len(Xt)) < 0.15
    m = lgb.LGBMClassifier(random_state=seed, **models.LGB_PARAMS)
    m.fit(Xt[~u], yt[~u], eval_set=[(Xt[u], yt[u])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


rows, info = [], []
for Y in (2013, 2014, 2015):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1]
    lgd = asof.loss_given_default(tr)
    pool = d[(d['issue_year'] == Y) & (d['term_m'] == 36)]
    print(Y, '36개월 발행 중 종결 비율', round(pool['resolved'].mean(), 4), flush=True)
    evi = pool.index[pool['resolved'].to_numpy()]
    P = [fit_lgb(X0.loc[tri, usable], ytr, s).predict_proba(X0.loc[evi, usable])[:, 1] for s in SEEDS]
    print(Y, '학습 완료', f'{(time.time() - t0) / 60:.1f}분', flush=True)
    cuts = {'1~2년 관측': f'{Y + 2}-01-01', '2~3년 관측': f'{Y + 3}-01-01', '전체': None}
    for name, T in cuts.items():
        keep = np.ones(len(evi), bool) if T is None else asof.known_at(d.loc[evi], T)
        ev = returns.attach_treasury(d.loc[evi[keep]]).reset_index(drop=True)
        r, tb = ev['r'].to_numpy(), ev['tb'].to_numpy()
        rnd = pd.concat([monthly(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)], axis=1).mean(axis=1)
        fe = pd.concat([monthly(ev, (1 - p[keep]) * r - p[keep] * lgd - tb) for p in P], axis=1).mean(axis=1) - rnd
        fp = pd.concat([monthly(ev, -p[keep]) for p in P], axis=1).mean(axis=1) - rnd
        rows.append(pd.DataFrame({'연도': Y, '관측': name, 'ym': fe.index, '기대수익 순위': fe.to_numpy(), '부도확률 순위': fp.to_numpy()}))
        info.append({'연도': Y, '관측': name, '남은 비율': keep.mean(), '부도율': ev['bad'].mean(), '평균 금리': ev['r'].mean(),
                     '평균 예측 PD': float(np.mean([p[keep].mean() for p in P]))})
        print(Y, name, f'{(time.time() - t0) / 60:.1f}분', flush=True)
    pd.concat(rows).to_csv(RESULTS / '26_resolution_monthly.csv', index=False)

M = pd.concat(rows)
M['차이'] = M['기대수익 순위'] - M['부도확률 순위']
rng = np.random.default_rng(2026)
out = []
for name, g in M.groupby('관측', sort=False):
    idx = rng.integers(0, len(g), size=(B, len(g)))
    rec = {'관측': name}
    for col in ['기대수익 순위', '부도확률 순위', '차이']:
        v = g[col].to_numpy()
        bs = v[idx].mean(axis=1)
        rec[col], rec[col + ' 하한'], rec[col + ' 상한'] = v.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)
    out.append(rec)
S = pd.DataFrame(out)
I = pd.DataFrame(info)
S.round(5).to_csv(RESULTS / '26_resolution_summary.csv', index=False)
I.round(4).to_csv(RESULTS / '26_resolution_info.csv', index=False)
pd.set_option('display.width', 220)
print(S.round(4).to_string(index=False))
print(M.groupby(['연도', '관측'], sort=False)[['기대수익 순위', '부도확률 순위', '차이']].mean().round(4).to_string())
print(I.round(3).to_string(index=False))
full, short = S.set_index('관측').loc['전체'], S.set_index('관측').loc['1~2년 관측']
if full['차이 하한'] <= 0:
    print('판정: 전체 표본에서도 기대수익 순위 우위가 불확실 -> C6 약화')
elif short['차이'] <= 0:
    print('판정: 종결 편향이 순위를 뒤집는다 -> 2017년 결과는 편향으로 설명됨')
else:
    print('판정: 자른 표본에서도 기대수익 순위 우위 유지 -> 2017년 결과는 종결 편향만으로 설명되지 않음')
print(f'총 {(time.time() - t0) / 60:.1f}분')
