"""34단계(검증): 아직 묻지 않은 근본 가정 6개.

A. 신호의 연도별 감소: 등급 중립 롱숏(상위 30% - 하위 30%)을 연도별로. test가 더 최근 대출이면 효과가 작을 수 있다.
   판정: 기대수익 점수와 계좌 개설 신호의 2016년 스프레드가 2013년의 절반 미만이면 '감소 뚜렷'.
B. 개인 투자자가 실제로 살 수 있었나: LC는 대출 일부를 기관용 통째 매각(initial_list_status = w)으로 돌렸다.
   개인이 살 수 있는 조각 대출(f)만으로도 롱숏 신호와 두 축 결론이 유지되나.
   판정: f만의 기대수익 롱숏 90% 하한 > 0 이고, f 안에서도 부도확률 순 Sharpe > 기대수익 순 Sharpe(승인 30%, 50%)면 '유지'.
C. 60개월 대출(2013~14년 발행, 거의 다 끝남): 모델 없이 등급별 Sharpe와 단순 신호 롱숏.
   판정: 등급별 Sharpe가 A에서 가장 높고 계좌 개설 롱숏 하한 > 0 이면 '60개월에도 유지'.
D. 위기 시기(2007~09년 발행 36개월): 모델 없이 등급별 평균 초과수익과 Sharpe. 침체기에 '안전한 등급 우위'가 더 커지나.
   판정: 위기 시기 A등급 Sharpe - E등급 Sharpe 차이가 2013~16년보다 크면 '침체기에 안정성 축이 더 중요'.
E. 투명한 단순 규칙: 등급마다 최근 24개월 개설 계좌가 적은 30%를 승인. 모델 없이도 되나.
   판정: 선별 가치(무작위 대비 평균 초과수익 차이)가 기대수익 순의 70% 이상이면 '단순 규칙으로 대부분 재현'.
F. LC 입장(가격 정보 없는 모델): 금리, 등급, 세부등급, 월 상환액을 뺀 모델로 학습. 과제가 '투자자'가 아니라 'LC의 승인'이라면 이 모델이 맞다.
   판정: 가격 정보 없는 모델의 AUC와 기대수익 선별 가치(이 모델 PD + 실제 금리로 계산)를 보고, 등급 중립 롱숏 하한 > 0 이면
   'LC가 매긴 가격 밖에도 정보가 있다'.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

sharpe = lambda x: x.mean() / x.std(ddof=1)
rng0 = np.random.default_rng(2026)


def boot_lo(x, b=2000, block=6):
    n = len(x)
    st = rng0.integers(0, max(1, n - block + 1), size=(b, int(np.ceil(n / block))))
    ix = np.concatenate([np.arange(block)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n]
    ix = np.minimum(ix, n - 1)
    bs = x[ix].mean(axis=1)
    return np.quantile(bs, 0.05), np.quantile(bs, 0.95)


def ls_spread(df, sig, frac=0.3):
    """발행월·등급마다 신호 상위 frac - 하위 frac의 묶음 IRR 차이, 발행월별 시계열."""
    g = df['grade'].astype(str).to_numpy()
    out = {}
    for ymv, ixm in df.groupby('ym').indices.items():
        L, S = [], []
        for gg in np.unique(g[ixm]):
            ix = ixm[g[ixm] == gg]
            s = sig[ix]
            ok = ~np.isnan(s)
            if ok.sum() < 10:
                continue
            lo, hi = np.nanquantile(s, frac), np.nanquantile(s, 1 - frac)
            L += list(ix[ok & (s >= hi)])
            S += list(ix[ok & (s <= lo)])
        if len(L) >= 20 and len(S) >= 20:
            out[ymv] = returns.pooled_irr(df.iloc[L], fee=FEE) - returns.pooled_irr(df.iloc[S], fee=FEE)
    return pd.Series(out).sort_index()


def grade_table(df):
    rows = []
    g = df['grade'].astype(str).to_numpy()
    tb = df['tb'].to_numpy()
    for gg in sorted(set(g)):
        ex = []
        for _, ixm in df.groupby('ym').indices.items():
            ix = ixm[g[ixm] == gg]
            if len(ix) >= 20:
                ex.append(returns.pooled_irr(df.iloc[ix], fee=FEE) - tb[ix].mean())
        ex = np.array(ex)
        if len(ex) >= 3:
            rows.append({'등급': gg, '발행월 수': len(ex), '평균 초과수익(%p)': 100 * ex.mean(), 'Sharpe': sharpe(ex),
                         '국채 이하 달': int((ex <= 0).sum())})
    return pd.DataFrame(rows)


out = {}
# ---------- A. 연도별 롱숏 ----------
sigA = {'기대수익 점수': ev['s_er'].to_numpy(), '계좌 개설(적을수록)': -ev['acc_open_past_24mths'].to_numpy(float),
        '연소득': ev['annual_inc'].to_numpy(float), 'FICO': ev['fico_range_low'].to_numpy(float)}
rowsA = []
for nm, s in sigA.items():
    sp = ls_spread(ev, s)
    yr = pd.Series(sp.values, index=[str(i)[:4] for i in sp.index])
    rowsA.append({'신호': nm, **{y: 100 * yr[y].mean() for y in ['2013', '2014', '2015', '2016']}})
A = pd.DataFrame(rowsA)
print('== A. 연도별 등급 중립 롱숏(%p)\n', A.round(2).to_string(index=False), flush=True)

# ---------- B. 조각 대출(f)만 ----------
f = (ev['initial_list_status'].astype(str) == 'f').to_numpy()
print('\n== B. 조각 대출(f) 비중 by 연도:', ev.groupby('issue_year')['initial_list_status'].apply(lambda s: (s.astype(str) == 'f').mean()).round(2).to_dict())
evf = ev[f].reset_index(drop=True)
spf = ls_spread(evf, evf['s_er'].to_numpy())
lo, hi = boot_lo(spf.to_numpy())
print(f'f만 기대수익 롱숏 {100 * spf.mean():.2f}%p, 90% 구간 {100 * lo:.2f}~{100 * hi:.2f}, 양수 달 {(spf > 0).mean():.2f}')
grp_f = evf.groupby('ym').indices
Pf, FVf, TBf = evf['funded_amnt'].to_numpy(float), evf['fv'].to_numpy(float), evf['tb'].to_numpy(float)


def mexcess_f(a):
    vals = []
    for m in sorted(grp_f):
        ix = grp_f[m]
        sel = ix[a[ix]]
        tb = TBf[ix].mean()
        w = Pf[sel].sum() / Pf[ix].sum()
        vals.append(w * (returns.pooled_irr(evf.iloc[sel], fee=FEE) - tb) if len(sel) else 0.0)
    return np.array(vals)


def top_f(score, q):
    a = np.zeros(len(evf), bool)
    for _, ix in evf.groupby('issue_year').indices.items():
        a[ix] = score[ix] >= np.quantile(score[ix], 1 - q)
    return a


for q in (0.3, 0.5):
    e1, e2 = mexcess_f(top_f(evf['s_er'].to_numpy(), q)), mexcess_f(top_f(-evf['p'].to_numpy(), q))
    print(f'f, 승인 {int(q * 100)}%: 기대수익 순 Sharpe {sharpe(e1):.2f} / 수익 {100 * e1.mean():.2f}%p, 부도확률 순 Sharpe {sharpe(e2):.2f} / 수익 {100 * e2.mean():.2f}%p')

# ---------- C. 60개월 ----------
d60 = d[d['issue_year'].isin([2013, 2014]) & (d['term_m'] == 60)]
print('\n== C. 60개월(2013~14) 종결 비율', round(d60['resolved'].mean(), 3))
e60 = returns.attach_treasury(d60[d60['resolved']]).reset_index(drop=True)
print(grade_table(e60).round(2).to_string(index=False))
sp60 = ls_spread(e60, -e60['acc_open_past_24mths'].to_numpy(float))
lo, hi = boot_lo(sp60.to_numpy())
print(f'60개월 계좌 개설 롱숏 {100 * sp60.mean():.2f}%p, 90% 구간 {100 * lo:.2f}~{100 * hi:.2f}, 양수 달 {(sp60 > 0).mean():.2f}', flush=True)

# ---------- D. 위기 시기 ----------
dc = d[d['issue_year'].isin([2007, 2008, 2009]) & (d['term_m'] == 36) & d['resolved']]
ec = returns.attach_treasury(dc).reset_index(drop=True)
print('\n== D. 위기 시기(2007~09, 36개월) 등급별, 대출', len(ec))
print(grade_table(ec).round(2).to_string(index=False))
print('비교: 2013~16 A Sharpe 5.24, E Sharpe 0.64 (33단계)')

# ---------- E. 단순 규칙 ----------
QE = [0.3, 0.5]
ac = ev['acc_open_past_24mths'].to_numpy(float)


def top_within_grade(score, q):
    a = np.zeros(len(ev), bool)
    for _, ix in ev.groupby(['issue_year', 'grade'], observed=True).indices.items():
        s = score[ix]
        ok = ~np.isnan(s)
        if ok.sum() == 0:
            continue
        a[ix[ok]] = s[ok] >= np.quantile(s[ok], 1 - q)
    return a


print('\n== E. 단순 규칙(등급 안 계좌 개설 적은 순)')
for q in QE:
    res = {}
    for nm, a in [('단순 규칙', top_within_grade(-ac, q)), ('기대수익 순', top(ev['s_er'].to_numpy(), q)),
                  ('부도확률 순', top(-ev['p'].to_numpy(), q))]:
        e = monthly_excess(a)[0]
        res[nm] = (e.mean(), sharpe(e))
    rnd = np.mean([monthly_excess(top(np.random.default_rng(100 + k).random(len(ev)), q))[0].mean() for k in range(5)])
    sv_rule, sv_er = res['단순 규칙'][0] - rnd, res['기대수익 순'][0] - rnd
    print(f'승인 {int(q * 100)}%: ' + ', '.join(f'{k} 수익 {100 * v[0]:.2f}%p Sharpe {v[1]:.2f}' for k, v in res.items())
          + f' | 선별 가치 단순 {100 * sv_rule:.2f} vs 기대수익 {100 * sv_er:.2f}%p -> 비율 {sv_rule / sv_er:.2f}')

# ---------- F. 가격 정보 없는 모델 ----------
PRICE = {'r', 'grade', 'sub_grade', 'installment', 'int_rate'}
print('\n== F. 가격 정보 없는 모델', flush=True)
pf, aucs = np.zeros(len(ev)), []
for Y in range(2013, 2017):
    tr = asof.training_set(d, f'{Y}-01-01')
    tri, ytr = tr.index, tr['bad'].to_numpy()
    usable = [c for c in feats if X0.loc[tri, c].nunique(dropna=True) > 1 and c not in PRICE]
    evi = d.index[(d['issue_year'] == Y) & d['resolved'] & (d['term_m'] == 36)]
    pY = np.mean([fit_lgb(X0.loc[tri, usable], ytr, s).predict_proba(X0.loc[evi, usable])[:, 1] for s in SEEDS], axis=0)
    m = (ev['issue_year'] == Y).to_numpy()
    pf[m] = pY
    aucs.append(roc_auc_score(ev.loc[m, 'bad'], pY))
    print(Y, f'AUC 가격 없음 {aucs[-1]:.3f} vs 기본 {roc_auc_score(ev.loc[m, "bad"], ev.loc[m, "p"]):.3f}', flush=True)
lgd_vec = ev['issue_year'].map({Y: asof.loss_given_default(asof.training_set(d, f'{Y}-01-01')) for Y in range(2013, 2017)}).to_numpy()
s_nf = (1 - pf) * ev['r'].to_numpy() - pf * lgd_vec - ev['tb'].to_numpy()
for q in QE:
    e_nf = monthly_excess(top(s_nf, q))[0]
    e_er = monthly_excess(top(ev['s_er'].to_numpy(), q))[0]
    print(f'승인 {int(q * 100)}%: 가격 없는 모델 수익 {100 * e_nf.mean():.2f}%p Sharpe {sharpe(e_nf):.2f} | 기본 {100 * e_er.mean():.2f}%p {sharpe(e_er):.2f}')
spnf = ls_spread(ev, -pf)
lo, hi = boot_lo(spnf.to_numpy())
print(f'가격 없는 모델 PD 등급 중립 롱숏 {100 * spnf.mean():.2f}%p, 90% 구간 {100 * lo:.2f}~{100 * hi:.2f}, 양수 달 {(spnf > 0).mean():.2f}')
print(f'총 {(time.time() - t0) / 60:.1f}분')
