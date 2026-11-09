"""14단계: 11단계에서 채택한 두 비교 방법(선별 곡선, 성과 귀속)을 엄격히 다시 검증한다.

미리 정한 검증 항목과 통과 기준
V1 시드 안정성: LightGBM 시드 5개에서 결론의 부호와 크기가 유지된다(선별 가치 > 0이 모든 시드, 연도별 3/4 이상).
V2 신뢰구간: 발행월 블록 부트스트랩(2,000회) 95% 구간이 0을 포함하지 않는다.
V3 플라시보: 무작위 점수끼리 비교한 선별 가치는 0 근처, 라벨을 섞어 학습한 모델은 '금리 효과'만 남는다.
V4 분해 단위: 성과 귀속을 등급(7)이 아니라 세부등급(35)으로 해도 '등급 안 선별'이 주된 효과로 남는다(상호작용 항 분리).
V5 보류 구간: 지금까지 쓰지 않은 2017년 발행분(종결 편향 있음)에서도 같은 방향이 나온다.
V6 부도확률 순위 주장: "부도확률 순위는 무작위와 비슷하다"가 시드와 연도에 걸쳐 유지된다.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, asof, models, evaluate, returns
from lc.config import RESULTS

raw = data.load()
d = data.parse(raw)
d = d[~d['policy_fail']]
pos = pd.Series(np.arange(len(d)), index=d.index)
X = data.model_frame(d, data.feature_columns(d))
y = d['bad'].to_numpy()
evm = (d['resolved'] & (d['term_m'] == 36)).to_numpy()
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2, 3, 4]
FEE = 0.01


def cohort_excess(ev, a):
    c = evaluate.cohort_irr(ev, a, fee=FEE)
    return (c['irr'] - c['tb'])


def curve_by_cohort(ev, score):
    """승인비율별, 발행월별 승인 대출 초과 IRR -> (발행월 x 승인비율) 표."""
    cols = {}
    for q in QS:
        cols[q] = cohort_excess(ev, score >= np.quantile(score, 1 - q))
    return pd.DataFrame(cols)


def brinson(ev, a, key):
    w = ev['funded_amnt'].to_numpy()
    bench = returns.pooled_irr(ev, fee=FEE)
    alloc = sel = inter = 0.0
    for g, ix in ev.groupby(key, observed=True).indices.items():
        wb = w[ix].sum() / w.sum()
        sa = ix[a[ix]]
        ws = w[sa].sum() / w[a].sum() if len(sa) else 0.0
        rb = returns.pooled_irr(ev.iloc[ix], fee=FEE)
        rs = returns.pooled_irr(ev.iloc[sa], fee=FEE) if len(sa) >= 30 else rb
        alloc += (ws - wb) * (rb - bench)
        sel += wb * (rs - rb)
        inter += (ws - wb) * (rs - rb)
    total = returns.pooled_irr(ev[a], fee=FEE) - bench
    return {'차이': total, '배분': alloc, '선별': sel, '상호작용': inter, '잔차': total - alloc - sel - inter}


rows, boot_cohort = [], []
for Y in range(2013, 2018):
    C = pd.Timestamp(f'{Y}-01-01')
    tr = asof.training_set(d, C)
    itr = pos[tr.index].to_numpy()
    iy = np.where((d['issue_year'] == Y).to_numpy())[0]
    ie = iy[evm[iy]]
    ev = returns.attach_treasury(d.iloc[ie]).reset_index(drop=True)
    lgd = asof.loss_given_default(tr)
    r, tb = ev['r'].to_numpy(), ev['tb'].to_numpy()
    rnd_curves = [curve_by_cohort(ev, np.random.default_rng(100 + k).random(len(ev))) for k in range(5)]
    rnd = sum(rnd_curves) / len(rnd_curves)
    # V3: 무작위 점수 하나를 나머지 평균과 비교 = 잡음 수준
    noise_sv = float((rnd_curves[0] - sum(rnd_curves[1:]) / 4).mean(axis=1).mean())
    scores = {}
    for s in SEEDS:
        m = models.fit_lgbm_pd(X.iloc[itr], y[itr], seed=s)
        p = m.predict_proba(X.iloc[ie])[:, 1]
        scores[f'기대수익 시드{s}'] = (1 - p) * r - p * lgd - tb
        scores[f'부도확률 시드{s}'] = -p
    ysh = np.random.default_rng(Y).permutation(y[itr])
    msh = models.fit_lgbm_pd(X.iloc[itr], ysh, seed=0)
    psh = msh.predict_proba(X.iloc[ie])[:, 1]
    scores['기대수익 라벨섞음'] = (1 - psh) * r - psh * lgd - tb
    for nm, sc in scores.items():
        cv = curve_by_cohort(ev, sc)
        per_cohort = (cv - rnd).mean(axis=1)  # 발행월별 선별 가치
        row = {'연도': Y, '점수': nm, '선별 가치': float(per_cohort.mean()), '무작위끼리 잡음': noise_sv}
        if nm.startswith('기대수익 시드') or nm == '기대수익 라벨섞음':
            a50 = sc >= np.quantile(sc, 0.5)
            for key, lab in [('grade', '등급'), ('sub_grade', '세부등급')]:
                b = brinson(ev, a50, key)
                for k, v in b.items():
                    row[f'{lab} {k}'] = v
        rows.append(row)
        if nm.startswith('기대수익 시드') or nm.startswith('부도확률 시드') or nm == '기대수익 라벨섞음':
            for ym, v in per_cohort.items():
                boot_cohort.append({'연도': Y, '점수': nm, 'ym': ym, '선별 가치': v})
    print(Y, '완료', flush=True)

R = pd.DataFrame(rows)
B = pd.DataFrame(boot_cohort)
RESULTS.mkdir(exist_ok=True)
R.round(5).to_csv(RESULTS / '14_validation_rows.csv', index=False)
B.round(6).to_csv(RESULTS / '14_validation_cohorts.csv', index=False)

# 요약: 시드 평균, 발행월 블록 부트스트랩 신뢰구간(2013~2016 평가 구간, 2017 보류 구간 따로)
B['종류'] = B['점수'].str.replace(r' 시드\d', '', regex=True)
rng = np.random.default_rng(0)
summ = []
for (kind, period), g in B.assign(구간=np.where(B['연도'] == 2017, '2017 보류', '2013~2016')).groupby(['종류', '구간']):
    by_m = g.groupby('ym')['선별 가치'].mean().to_numpy()  # 시드 평균 후 발행월 단위
    bs = [rng.choice(by_m, len(by_m)).mean() for _ in range(2000)]
    summ.append({'점수': kind, '구간': period, '발행월 수': len(by_m), '선별 가치': by_m.mean(),
                 '95% 하한': np.percentile(bs, 2.5), '95% 상한': np.percentile(bs, 97.5)})
S = pd.DataFrame(summ)
S.round(5).to_csv(RESULTS / '14_validation_ci.csv', index=False)
seed_tab = R[R['점수'].str.startswith('기대수익 시드') | R['점수'].str.startswith('부도확률 시드')].copy()
seed_tab['종류'] = seed_tab['점수'].str.replace(r' 시드\d', '', regex=True)
spread = seed_tab.groupby(['종류', '연도'])['선별 가치'].agg(['mean', 'min', 'max']).round(5)
br = R[R['점수'].str.startswith('기대수익')].copy()
br['종류'] = np.where(br['점수'] == '기대수익 라벨섞음', '라벨섞음', '실제 모델(시드 평균)')
bcols = [c for c in br.columns if c.startswith('등급 ') or c.startswith('세부등급 ')]
BR = br.groupby(['종류', '연도'])[bcols].mean().round(5)
BR.to_csv(RESULTS / '14_validation_brinson.csv')
pd.set_option('display.width', 250)
print(S.round(5).to_string(index=False))
print(spread.to_string())
print(BR.to_string())
print('무작위끼리 잡음(연도별)', R.groupby('연도')['무작위끼리 잡음'].first().round(5).to_dict())
