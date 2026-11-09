"""15단계: 공식 test 평가. docs/test_protocol.md(v1)를 그대로 실행한다.

사용:
  python scripts/15_test_evaluate.py <test 파일(.csv 또는 .parquet)>
  python scripts/15_test_evaluate.py --dry-run   # train의 30%를 가짜 test로 떼어 예행연습
결과: results/test/ (예행연습은 results/test_dryrun/) 에 CSV와 summary.md
"""
import sys
import zlib
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
from lc import data, asof, models, evaluate, returns, benchmarks
from lc.config import RESULTS

DRY = '--dry-run' in sys.argv
FEE = 0.01
SEEDS = [0, 1, 2, 3, 4]
YEARS_MAIN = [2013, 2014, 2015, 2016]
YEARS_EXTRA = [2017, 2018]
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OUT = RESULTS / ('test_dryrun' if DRY else 'test')
OUT.mkdir(parents=True, exist_ok=True)
log_lines = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# ---------- 0. 데이터 ----------
train_raw = data.load()
if DRY:
    key = train_raw['id'].astype(str).map(lambda s: zlib.crc32(s.encode()) % 10)
    test_raw = train_raw[key < 3].copy()
    train_raw = train_raw[key >= 3].copy()
    log('예행연습: train의 30%(id 해시 기준)를 가짜 test로 사용')
else:
    p = Path([a for a in sys.argv[1:] if not a.startswith('--')][0])
    test_raw = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p, low_memory=False)
    test_raw = test_raw[test_raw['loan_status'].notna()].copy()
log('train', len(train_raw), '| test', len(test_raw))

# ---------- 1. 점검 ----------
need = {'total_pymnt', 'last_pymnt_d', 'loan_status'}
has_post = need <= set(test_raw.columns)
overlap = test_raw['id'].astype(str).isin(set(train_raw['id'].astype(str)))
log('상환 결과 컬럼 있음:', has_post, '| train과 id 중복:', int(overlap.sum()))
test_raw = test_raw[~overlap.to_numpy()]
st = data.classify(test_raw['loan_status']).value_counts()
log('test 결과 상태:', st.to_dict())

tr_all = data.parse(train_raw.reset_index(drop=True))
tr_all = tr_all[~tr_all['policy_fail']]
te_all = data.parse(test_raw.reset_index(drop=True))
te_all = te_all[~te_all['policy_fail']]
feats = data.feature_columns(tr_all)
missing = [c for c in feats if c not in te_all.columns]
assert not missing, f'test에 없는 특성: {missing}'
both = pd.concat([tr_all[feats], te_all[feats]], keys=['tr', 'te'])
Xb = data.model_frame(both, feats)  # 범주를 train·test에 같게 맞춘다
Xtr, Xte = Xb.loc['tr'], Xb.loc['te']
Xtr.index, Xte.index = tr_all.index, te_all.index
ytr = tr_all['bad'].to_numpy()

# ---------- 2. 시점 학습 예측(시드 5개 평균) ----------
P = pd.Series(np.nan, index=te_all.index)
LGD = {}
for Y in YEARS_MAIN + YEARS_EXTRA:
    C = pd.Timestamp(f'{Y}-01-01')
    tr = asof.training_set(tr_all, C)
    idx = te_all.index[te_all['issue_year'] == Y]
    if len(idx) == 0:
        continue
    ps = [models.fit_lgbm_pd(Xtr.loc[tr.index], ytr[tr_all.index.get_indexer(tr.index)], seed=s).predict_proba(Xte.loc[idx])[:, 1]
          for s in SEEDS]
    P[idx] = np.mean(ps, axis=0)
    LGD[Y] = asof.loss_given_default(tr)
    log(f'{Y}: 학습 {len(tr):,}건, 예측 {len(idx):,}건, LGD {LGD[Y]:.3f}')

# 참고용: 전체 train으로 한 번에 학습(미래 정보 포함)
res_tr = tr_all[tr_all['resolved']]
P_full = pd.Series(np.mean([models.fit_lgbm_pd(Xtr.loc[res_tr.index], res_tr['bad'].to_numpy(), seed=s).predict_proba(Xte)[:, 1]
                            for s in SEEDS], axis=0), index=te_all.index)
LGD_full = float(1 - (res_tr['total_pymnt'] / res_tr['funded_amnt'])[res_tr['bad'] == 1].mean())

# 승인 목록(제출용, R1) — 예측이 있는 모든 test 대출
te_all['pd_asof'] = P
te_all['lgd_asof'] = te_all['issue_year'].map(LGD)
te_tb = returns.attach_treasury(te_all[te_all['issue_year'].between(2007, 2020)])
er_all = (1 - te_tb['pd_asof']) * te_tb['r'] - te_tb['pd_asof'] * te_tb['lgd_asof']
pd.DataFrame({'id': test_raw.reset_index(drop=True).loc[te_tb.index, 'id'], 'issue_year': te_tb['issue_year'],
              'pd': te_tb['pd_asof'], 'expected_return': er_all, 'approve_R1': (er_all > te_tb['tb']) & te_tb['pd_asof'].notna()}) \
    .to_csv(OUT / 'approval_list_R1.csv', index=False)

if not has_post:
    log('상환 결과 컬럼이 없어 수익 평가는 하지 않음(프로토콜 1절)')
    (OUT / 'summary.md').write_text('\n'.join(log_lines) + '\n')
    sys.exit(0)


# ---------- 3. 평가 함수 ----------
def frame(years, term, pcol, lgd_map=None, lgd_const=None):
    m = te_all['resolved'] & te_all['issue_year'].isin(years) & (te_all['term_m'] == term) & te_all[pcol].notna()
    e = returns.attach_treasury(te_all[m]).reset_index(drop=True)
    e['baa'] = benchmarks.benchmark(e, '회사채 BAA')
    lg = e['issue_year'].map(lgd_map).to_numpy() if lgd_map is not None else lgd_const
    e['er'] = (1 - e[pcol]) * e['r'] - e[pcol] * lg
    return e


def rules(e):
    yrs = e['issue_year'].to_numpy()
    top50, low50 = np.zeros(len(e), bool), np.zeros(len(e), bool)
    for y in np.unique(yrs):
        ix = np.where(yrs == y)[0]
        top50[ix] = e['er'].to_numpy()[ix] >= np.quantile(e['er'].to_numpy()[ix], 0.5)
        low50[ix] = e['pdx'].to_numpy()[ix] <= np.quantile(e['pdx'].to_numpy()[ix], 0.5)
    return {'R0 전부 승인': np.ones(len(e), bool), 'R1 기대수익>국채': e['er'].to_numpy() > e['tb'].to_numpy(),
            'R2 기대수익 상위 50%': top50, 'R3 부도확률 하위 50%': low50}


def rule_table(e):
    rows = []
    for nm, a in rules(e).items():
        s = evaluate.summarize(e, a, fee=FEE)
        cb = evaluate.cohort_irr(e.assign(tb=e['baa']), a, fee=FEE)
        rows.append({'규칙': nm, '승인비율': s['승인비율'], '코호트 IRR': s['코호트_평균IRR'], '국채 대비 초과수익': s['코호트_초과수익'],
                     '국채 이하 발행월': s['국채이하_코호트수'], '회사채 대비 초과수익': (cb['irr'] - cb['tb']).mean(),
                     '회사채 이하 발행월': int((cb['irr'] < cb['tb']).sum()), '발행월 수': len(cb), '승인 부도율': s['승인_부도율'],
                     '하위5% 평균': s['하위5%_평균(CVaR)'], '과제식 Sharpe': s['과제식_Sharpe(부도=0, rf5%)']})
    return pd.DataFrame(rows)


def brinson(e, a, key):
    w = e['funded_amnt'].to_numpy()
    bench = returns.pooled_irr(e, fee=FEE)
    alloc = sel = inter = 0.0
    for _, ix in e.groupby(key, observed=True).indices.items():
        wb = w[ix].sum() / w.sum()
        sa = ix[a[ix]]
        ws = w[sa].sum() / w[a].sum() if len(sa) else 0.0
        rb = returns.pooled_irr(e.iloc[ix], fee=FEE)
        rs = returns.pooled_irr(e.iloc[sa], fee=FEE) if len(sa) >= 30 else rb
        alloc += (ws - wb) * (rb - bench)
        sel += wb * (rs - rb)
        inter += (ws - wb) * (rs - rb)
    return {'차이': returns.pooled_irr(e[a], fee=FEE) - bench, '배분': alloc, '선별': sel, '상호작용': inter}


def selection_value(e, score, n_random=20):
    """발행월별 선별 가치(승인비율 10~90% 평균, 무작위 20회 평균 대비) -> 발행월 Series."""
    def curve(s):
        return pd.DataFrame({q: (lambda c: c['irr'] - c['tb'])(evaluate.cohort_irr(e, s >= np.quantile(s, 1 - q), fee=FEE)) for q in QS})
    rnd = sum(curve(np.random.default_rng(1000 + k).random(len(e))) for k in range(n_random)) / n_random
    return (curve(score) - rnd).mean(axis=1)


def md_table(df):
    """외부 패키지 없이 마크다운 표로 바꾼다."""
    cols = [str(c) for c in df.columns]
    fmt = lambda v: f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def boot_ci(v, n=2000):
    rng = np.random.default_rng(0)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------- 4. 실행 ----------
te_all['pdx'] = te_all['pd_asof']
main = frame(YEARS_MAIN, 36, 'pd_asof', LGD)
log('주 평가 표본(2013~16, 36개월 종결):', len(main), '건, 발행월', main['ym'].nunique())
T2 = rule_table(main)
T2.round(5).to_csv(OUT / 'T2_rules_main.csv', index=False)

R2 = rules(main)['R2 기대수익 상위 50%']
br = []
for y in YEARS_MAIN + ['통합']:
    m = np.ones(len(main), bool) if y == '통합' else (main['issue_year'] == y).to_numpy()
    sub = main[m].reset_index(drop=True)
    for key, lab in [('grade', '등급 7개'), ('sub_grade', '세부등급 35개')]:
        br.append({'연도': y, '분해 단위': lab, **brinson(sub, R2[m], key)})
T3 = pd.DataFrame(br)
T3.round(5).to_csv(OUT / 'T3_brinson.csv', index=False)

sv_er = selection_value(main, main['er'].to_numpy())
sv_pd = selection_value(main, -main['pdx'].to_numpy())
T4 = pd.DataFrame([{'순위 기준': '기대수익', **dict(zip(['선별 가치', '95% 하한', '95% 상한'], boot_ci(sv_er.to_numpy())))},
                   {'순위 기준': '부도확률(낮은 순)', **dict(zip(['선별 가치', '95% 하한', '95% 상한'], boot_ci(sv_pd.to_numpy())))}])
T4.round(5).to_csv(OUT / 'T4_selection_value.csv', index=False)

extra = []
for lab, yrs, term in [('2013~16 60개월', YEARS_MAIN, 60), ('2017~18 36개월(종결 편향)', YEARS_EXTRA, 36)]:
    e = frame(yrs, term, 'pd_asof', LGD)
    if len(e) > 1000:
        t = rule_table(e)
        t.insert(0, '표본', lab)
        extra.append(t)
T5 = pd.concat(extra) if extra else pd.DataFrame()
T5.round(5).to_csv(OUT / 'T5_rules_extra.csv', index=False)

te_all['pd_full'] = P_full
te_all['pdx'] = te_all['pd_full']
full = frame(YEARS_MAIN, 36, 'pd_full', lgd_const=LGD_full)
T6 = rule_table(full)
T6.round(5).to_csv(OUT / 'T6_rules_full_train_reference.csv', index=False)

# ---------- 5. 가설 판정(4절 기준 그대로) ----------
pooled = T3[(T3['연도'] == '통합') & (T3['분해 단위'] == '세부등급 35개')].iloc[0]
yearly = T3[(T3['연도'] != '통합') & (T3['분해 단위'] == '세부등급 35개')]
h1 = (pooled['선별'] > 0) and (pooled['선별'] > pooled['배분']) and ((yearly['선별'] > 0).sum() >= 3)
h2 = T4.loc[0, '95% 하한'] > 0
r0 = T2[T2['규칙'] == 'R0 전부 승인'].iloc[0]
h3 = (r0['발행월 수'] - r0['국채 이하 발행월']) / r0['발행월 수'] >= 0.9
r1 = T2[T2['규칙'] == 'R1 기대수익>국채'].iloc[0]
h4 = r1['회사채 이하 발행월'] < r0['회사채 이하 발행월']
verdict = pd.DataFrame([('H1 성과 귀속', h1), ('H2 선별 가치', h2), ('H3 국채 대비', h3), ('H4 회사채 대비', h4)], columns=['가설', '지지'])
verdict.to_csv(OUT / 'verdict.csv', index=False)

pd.set_option('display.width', 250)
md = ['# test 평가 결과' + (' (예행연습)' if DRY else ''), '', '프로토콜: docs/test_protocol.md v1', '', '## 점검', *log_lines, '',
      '## 가설 판정', md_table(verdict), '', '## 규칙별 성과(주 표본)', md_table(T2), '',
      '## 성과 귀속', md_table(T3), '', '## 선별 가치', md_table(T4), '',
      '## 보조 표본', md_table(T5) if len(T5) else '(없음)', '',
      '## 참고: 전체 train 학습 모델(미래 정보 포함)', md_table(T6)]
(OUT / 'summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
