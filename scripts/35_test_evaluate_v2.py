"""test 평가 v2: docs/test_protocol_v2.md 의 고정 규칙 그대로 실행한다.

사용:
  python scripts/35_test_evaluate_v2.py <test 파일>   # 실제 test
  python scripts/35_test_evaluate_v2.py --dry-run      # train의 30%를 가짜 test로 떼어 예행연습
v1(15_test_evaluate.py)의 점검·예측·표 T2~T6·H1~H4를 그대로 먼저 계산하고, v2의 규칙·지표·H5~H8을 덧붙인다.
결과: results/test_v2/ (예행연습은 results/test_v2_dryrun/)
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '15_test_evaluate.py').read_text()
SRC = SRC.replace("RESULTS / ('test_dryrun' if DRY else 'test')", "RESULTS / ('test_v2_dryrun' if DRY else 'test_v2')")
SRC = SRC.replace("pd.set_option('display.width', 250)", "V1_DONE = True\npd.set_option('display.width', 250)")
exec(compile(SRC[:SRC.index("(OUT / 'summary.md').write_text('\\n'.join(md)")], '15_test_evaluate.py', 'exec'))
assert V1_DONE

# ===================== v2 =====================
QV = [0.3, 0.5]
BLOCK, NB = 6, 2000
sharpe = lambda x: float(np.mean(x) / np.std(x, ddof=1))
jit_rng = np.random.default_rng(11)


def block_ci(x, alpha=0.10):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(2026)
    st = rng.integers(0, max(1, n - BLOCK + 1), size=(NB, int(np.ceil(n / BLOCK))))
    ix = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n], n - 1)
    bs = x[ix].mean(axis=1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def by_year_top(e, score, q):
    a = np.zeros(len(e), bool)
    for _, ix in e.groupby('issue_year').indices.items():
        a[ix] = score[ix] >= np.quantile(score[ix], 1 - q)
    return a


def within_grade_top(e, score, q):
    a = np.zeros(len(e), bool)
    for _, ix in e.groupby(['issue_year', 'grade'], observed=True).indices.items():
        s = score[ix]
        ok = ~np.isnan(s)
        if ok.sum():
            a[ix[ok]] = s[ok] >= np.quantile(s[ok], 1 - q)
    return a


def v2_rules(e, q):
    jit = jit_rng.random(len(e)) * 1e-6
    er, pdv = e['er'].to_numpy(), e['pdx'].to_numpy()
    acc = e['acc_open_past_24mths'].to_numpy(float) if 'acc_open_past_24mths' in e else np.full(len(e), np.nan)
    return {'V1 기대수익 순': by_year_top(e, er + jit, q), 'V2 부도확률 낮은 순': by_year_top(e, -pdv + jit, q),
            'V3 등급 안 계좌 개설 적은 순': within_grade_top(e, -acc + jit, q),
            'V4 등급 안 부도확률 낮은 순': within_grade_top(e, -pdv + jit, q)}


def monthly(e, a):
    """발행월별 (승인 묶음 IRR - 국채)와 포트폴리오 초과수익 = 승인 금액 비중 × 그 값."""
    P = e['funded_amnt'].to_numpy(float)
    tb = e['tb'].to_numpy(float)
    pool, port = [], []
    for _, ix in sorted(e.groupby('ym').indices.items()):
        sel = ix[a[ix]]
        t = tb[ix].mean()
        if len(sel) == 0:
            pool.append(np.nan)
            port.append(0.0)
            continue
        r = returns.pooled_irr(e.iloc[sel], fee=FEE) - t
        pool.append(r)
        port.append(P[sel].sum() / P[ix].sum() * r)
    return np.array(pool), np.array(port)


def metrics(e, a, R_loan, x_loan, npv):
    pool, port = monthly(e, a)
    xa = x_loan[a]
    return {'실제 승인비율': a.mean(), '승인 묶음 초과수익': np.nanmean(pool), '포트폴리오 초과수익': port.mean(),
            '포트폴리오 Sharpe': sharpe(port), '건별 Sharpe(승인만)': sharpe(xa), '건별 Sharpe(거절=0)': sharpe(np.where(a, x_loan, 0.0)),
            '하위5% 평균': np.sort(xa)[: max(1, int(len(xa) * 0.05))].mean(), '총 NPV(백만$)': npv[a].sum() / 1e6}


def loan_level(e):
    R = returns.reinvested_return(e, e['tb'].to_numpy()).to_numpy()
    fv = e['funded_amnt'].to_numpy(float) * (1 + R) ** (e['term_m'].to_numpy() / 12) * (1 - FEE)
    R_net = (fv / e['funded_amnt'].to_numpy(float)) ** (12 / e['term_m'].to_numpy()) - 1
    npv = fv / (1 + e['tb'].to_numpy()) ** (e['term_m'].to_numpy() / 12) - e['funded_amnt'].to_numpy(float)
    return R_net, R_net - e['tb'].to_numpy(), npv


def v2_table(e, label):
    R, x, npv = loan_level(e)
    rows = []
    for q in QV:
        rs = v2_rules(e, q)
        for nm, a in rs.items():
            rows.append({'표본': label, '목표 승인비율': q, '규칙': nm, **metrics(e, a, R, x, npv)})
        rnd = [metrics(e, by_year_top(e, np.random.default_rng(500 + k).random(len(e)), q), R, x, npv) for k in range(10)]
        rows.append({'표본': label, '목표 승인비율': q, '규칙': 'V0 무작위(10회 평균)', **pd.DataFrame(rnd).mean().to_dict()})
    return pd.DataFrame(rows)


def ls_spread(e, sig, frac=0.3):
    g = e['grade'].astype(str).to_numpy()
    out = []
    for _, ixm in sorted(e.groupby('ym').indices.items()):
        L, S = [], []
        for gg in np.unique(g[ixm]):
            ix = ixm[g[ixm] == gg]
            s = sig[ix] + jit_rng.random(len(ix)) * 1e-6
            ok = ~np.isnan(s)
            if ok.sum() < 10:
                continue
            k = max(1, int(ok.sum() * frac))
            order = ix[ok][np.argsort(s[ok])]
            S += list(order[:k])
            L += list(order[-k:])
        if len(L) >= 20 and len(S) >= 20:
            out.append(returns.pooled_irr(e.iloc[L], fee=FEE) - returns.pooled_irr(e.iloc[S], fee=FEE))
    return np.array(out)


def ls_table(e, label):
    sigs = {'기대수익 점수': e['er'].to_numpy(), '부도확률(낮을수록)': -e['pdx'].to_numpy()}
    if 'acc_open_past_24mths' in e:
        sigs['계좌 개설(적을수록)'] = -e['acc_open_past_24mths'].to_numpy(float)
    sigs['무작위(위약)'] = np.random.default_rng(7).random(len(e))
    rows = []
    for nm, s in sigs.items():
        sp = ls_spread(e, s)
        if len(sp) < 6:
            continue
        lo, hi = block_ci(sp)
        rows.append({'표본': label, '신호': nm, '롱-숏': sp.mean(), '90% 하한': lo, '90% 상한': hi, '양수 달 비율': (sp > 0).mean(), '발행월 수': len(sp)})
    return pd.DataFrame(rows)


def grade_table(e, label):
    rows = []
    g = e['grade'].astype(str).to_numpy()
    tb = e['tb'].to_numpy()
    for gg in sorted(set(g)):
        ex = []
        for _, ixm in e.groupby('ym').indices.items():
            ix = ixm[g[ixm] == gg]
            if len(ix) >= 20:
                ex.append(returns.pooled_irr(e.iloc[ix], fee=FEE) - tb[ix].mean())
        if len(ex) >= 3:
            rows.append({'표본': label, '등급': gg, '발행월 수': len(ex), '평균 초과수익': np.mean(ex), 'Sharpe': sharpe(ex)})
    return pd.DataFrame(rows)


# 만기 도달 비율: 발행월 + 만기가 데이터 마지막 납부월 이전인 대출 비율(진행 중 대출이 없을 때 종결 편향 점검)
last_obs = pd.to_datetime(te_all['last_pymnt_d'], format='%b-%Y', errors='coerce').max()
issue_dt = pd.to_datetime(te_all['issue_d'], format='%b-%Y', errors='coerce')
matured = (issue_dt + pd.to_timedelta(te_all['term_m'] * 30.44, unit='D')) <= last_obs
Y0 = te_all.assign(matured=matured.to_numpy()).groupby(['issue_year', 'term_m'])['matured'].mean().unstack()
Y0.round(3).to_csv(OUT / 'V0_maturity_ratio.csv')
log('만기 도달 비율(발행연도 × 만기):', Y0.round(2).to_dict())

te_all['pdx'] = te_all['pd_asof']
main = frame(YEARS_MAIN, 36, 'pd_asof', LGD)
samples = [(main, '주 표본 2013~16 36개월')]
e60 = frame(YEARS_MAIN, 60, 'pd_asof', LGD)
if len(e60) > 1000:
    samples.append((e60, '보조 2013~16 60개월'))
for y in YEARS_EXTRA:
    ey = frame([y], 36, 'pd_asof', LGD)
    if len(ey) > 1000:
        samples.append((ey, f'보조 {y} 36개월(종결 편향 가능)'))

V1t = pd.concat([v2_table(e, lab) for e, lab in samples])
V2t = pd.concat([ls_table(e, lab) for e, lab in samples])
V3t = pd.concat([grade_table(e, lab) for e, lab in samples])
yrs = []
for y in YEARS_MAIN:
    ey = main[main['issue_year'] == y].reset_index(drop=True)
    if len(ey) > 1000:
        yrs.append(ls_table(ey, f'{y} 36개월'))
V4t = pd.concat(yrs) if yrs else pd.DataFrame()
for nm, t in [('V1_rules_metrics', V1t), ('V2_longshort', V2t), ('V3_grade_sharpe', V3t), ('V4_longshort_by_year', V4t)]:
    t.round(5).to_csv(OUT / f'{nm}.csv', index=False)

# ---------- v2 가설 판정(프로토콜 v2 4절) ----------
m1 = V1t[V1t['표본'] == '주 표본 2013~16 36개월'].set_index(['목표 승인비율', '규칙'])
ls_main = V2t[V2t['표본'] == '주 표본 2013~16 36개월'].set_index('신호')
h5 = bool(ls_main.loc['기대수익 점수', '90% 하한'] > 0)
h6 = all(m1.loc[(q, 'V2 부도확률 낮은 순'), '포트폴리오 Sharpe'] > m1.loc[(q, 'V1 기대수익 순'), '포트폴리오 Sharpe']
         and m1.loc[(q, 'V1 기대수익 순'), '포트폴리오 초과수익'] > m1.loc[(q, 'V2 부도확률 낮은 순'), '포트폴리오 초과수익'] for q in QV)
h7 = all(m1.loc[(q, 'V3 등급 안 계좌 개설 적은 순'), '포트폴리오 Sharpe'] > m1.loc[(q, 'V2 부도확률 낮은 순'), '포트폴리오 Sharpe']
         and m1.loc[(q, 'V3 등급 안 계좌 개설 적은 순'), '포트폴리오 초과수익'] > m1.loc[(q, 'V2 부도확률 낮은 순'), '포트폴리오 초과수익'] for q in QV)
h8 = all(m1.loc[(0.3, 'V2 부도확률 낮은 순'), c] > m1.loc[(0.3, 'V1 기대수익 순'), c]
         for c in ['포트폴리오 Sharpe', '건별 Sharpe(승인만)', '건별 Sharpe(거절=0)'])
placebo_ok = bool(ls_main.loc['무작위(위약)', '90% 하한'] <= 0 <= ls_main.loc['무작위(위약)', '90% 상한'])
verdict2 = pd.DataFrame([('H5 등급 안 신호', h5), ('H6 두 축(수익 V1, 안정성 V2)', h6), ('H7 단순 규칙이 V2를 두 축 모두 이김', h7),
                         ('H8 세 Sharpe 정의 모두 V2 > V1', h8), ('점검: 위약 구간이 0 포함', placebo_ok)], columns=['가설', '지지'])
verdict2.to_csv(OUT / 'verdict_v2.csv', index=False)

md += ['', '# v2 추가', '프로토콜: docs/test_protocol_v2.md', '', '## 만기 도달 비율(발행연도 × 만기)', md_table(Y0.reset_index()), '',
       '## v2 가설 판정', md_table(verdict2), '', '## 규칙별 수익·안정성 지표', md_table(V1t), '',
       '## 등급 중립 롱숏', md_table(V2t), '', '## 연도별 롱숏(주 표본)', md_table(V4t) if len(V4t) else '(없음)', '',
       '## 등급별 Sharpe', md_table(V3t)]
(OUT / 'summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md[-12:]))
