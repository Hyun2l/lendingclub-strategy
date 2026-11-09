"""LC 증거 실험 묶음. 사용: python run_all.py [--smoke]
A 경쟁위험 누적발생률(Current 편향)  B 수익률 분포/왜도  E 정보 기여(LC 가격 vs 나머지)
D 가상 스냅샷 백테스트(그 시점 정보만 사용)  C 반복 분할(목표변수·규칙·위험지표·플라시보)
결과는 이 폴더의 csv로 단계별 저장. C는 시드별로 이어서 실행됨."""
import warnings, time, sys, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

SMOKE = '--smoke' in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'smoke' if SMOKE else 'results'); os.makedirs(OUT, exist_ok=True)
P = os.path.join(HERE, '..', '..', 'data', 'processed', 'lc_train.parquet')
TREES = 30 if SMOKE else 1500
NSEEDS = 2 if SMOKE else 20
t0 = time.time()
def log(*a): print(f'[{(time.time()-t0)/60:6.1f}분]', *a, flush=True)

# ---------- 준비 ----------
raw = pd.read_parquet(P); raw = raw[raw.loan_status.notna()].copy()
if SMOKE: raw = raw.sample(frac=0.06, random_state=1)
raw['ym'] = pd.to_datetime(raw.issue_d, format='%b-%Y'); raw['y'] = raw.ym.dt.year
raw['r'] = pd.to_numeric(raw.int_rate.astype('string').str.strip().str.rstrip('%')).astype(float) / 100
raw['Tm'] = raw.term.astype('string').str.extract(r'(\d+)')[0].astype(int)
lp = pd.to_datetime(raw.last_pymnt_d, format='%b-%Y', errors='coerce')
tau = ((lp.dt.year - raw.ym.dt.year) * 12 + lp.dt.month - raw.ym.dt.month).fillna(1).clip(1, None)
raw['tau'] = np.minimum(tau, raw.Tm + 12).astype(int)
raw['ev'] = np.select([raw.loan_status.isin(['Charged Off', 'Default']), raw.loan_status.eq('Fully Paid')], [1, 2], 0)
raw['res'] = raw.ev > 0
n = np.minimum(raw.tau, raw.Tm).astype(float); moic = (raw.total_pymnt / raw.funded_amnt).astype(float)
def fvret(rf):
    i = (1 + rf) ** (1 / 12) - 1
    fv = (moic / n) * ((1 + i) ** raw.Tm - (1 + i) ** (raw.Tm - n)) / i
    return np.clip(fv, 1e-3, None) ** (12 / raw.Tm) - 1
raw['RC'] = fvret(.05); raw['RD'] = fvret(.02)
raw['RA'] = np.where(raw.ev == 2, raw.r, 0.0)
raw['pratio'] = (raw.total_pymnt / (raw.Tm * raw.installment)).astype(float)
raw['bad'] = (raw.ev == 1).astype('int8')
for c in ['earliest_cr_line', 'sec_app_earliest_cr_line']:
    raw[c] = pd.to_datetime(raw[c], format='%b-%Y', errors='coerce').dt.year.astype('float32')
raw['revol_util'] = pd.to_numeric(raw.revol_util.astype('string').str.rstrip('%'), errors='coerce')
EX = ['id', 'loan_status', 'issue_d', 'int_rate', 'term', 'url', 'emp_title', 'title', 'zip_code', 'policy_code', 'pymnt_plan',
      'next_pymnt_d', 'last_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'debt_settlement_flag',
      'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
      'collection_recovery_fee', 'last_pymnt_amnt', 'out_prncp', 'out_prncp_inv', 'deferral_term', 'payment_plan_start_date',
      'orig_projected_additional_accrued_interest', 'ym', 'y', 'tau', 'ev', 'res', 'RC', 'RD', 'RA', 'pratio', 'bad']
EX += [c for c in raw.columns if c.startswith(('hardship_', 'settlement_'))]
F = [c for c in raw.columns if c not in EX]
for c in F:
    if pd.api.types.is_numeric_dtype(raw[c]): raw[c] = raw[c].astype('float32')
    else: raw[c] = raw[c].astype('category')
d = raw[raw.res].copy()
log('준비 완료', raw.shape, '종결', d.shape, '특성', len(F))

def sharpe(e, a):
    x = np.where(a, e, 0.0); s = x.std(ddof=1); return x.mean() / s if s > 0 else 0.0
def fit(Xtr, y, Xva, yv, kind, trees=None):
    k = dict(n_estimators=trees or TREES, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
             min_child_samples=200, reg_lambda=5, cat_smooth=50, n_jobs=-1, verbose=-1)
    m = (lgb.LGBMClassifier if kind == 'bin' else lgb.LGBMRegressor)(**k)
    return m.fit(Xtr, y, eval_set=[(Xva, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
def pick(s, e):
    best, th = -9, None
    for q in np.linspace(0, .98, 50):
        t = np.quantile(s, q); v = sharpe(e, s >= t)
        if v > best: best, th = v, t
    return th

# ---------- A: 경쟁위험 누적발생률 ----------
def cif(g, H):
    S, c1, c2 = 1.0, 0.0, 0.0
    t, e = g.tau.values, g.ev.values
    for m in range(1, H + 1):
        nr = (t >= m).sum()
        if nr == 0: return np.nan, np.nan, 0
        d1 = ((t == m) & (e == 1)).sum(); d2 = ((t == m) & (e == 2)).sum()
        c1 += S * d1 / nr; c2 += S * d2 / nr; S *= 1 - (d1 + d2) / nr
    return c1, c2, nr
rows = []
for (y, T), g in raw[raw.y.between(2012, 2019)].groupby(['y', 'Tm']):
    T = int(T); y = int(y); H = int({2012: T, 2013: T}.get(y, min(T, max(6, (2020 - y) * 12 - 3))))
    c1, c2, nr = cif(g, H); rs = g[g.res]
    rows.append(dict(발행연도=y, 만기=T, 건수=len(g), 종결비율=g.res.mean(), 종결표본_부도율=rs.bad.mean(), 전체중_확정부도비율=g.bad.mean(),
                     관측개월=H, 누적부도_CIF=c1, 누적상환_CIF=c2, 그시점_위험집합=nr))
pd.DataFrame(rows).round(4).to_csv(f'{OUT}/A_cif.csv', index=False); log('A 완료')

# ---------- B: 수익률 분포 ----------
rows = []
for gname, g in d.groupby('grade', observed=True):
    x = g.RC.values
    rows.append(dict(등급=gname, 건수=len(x), 평균=x.mean(), 표준편차=x.std(), 왜도=skew(x), 하방편차=np.sqrt((np.minimum(x - .05, 0) ** 2).mean()),
                     하위5퍼센트=np.quantile(x, .05), 부도율=g.bad.mean(), 평균금리=g.r.mean()))
pd.DataFrame(rows).round(4).to_csv(f'{OUT}/B_dist.csv', index=False); log('B 완료')

# ---------- E: 정보 기여 ----------
PRICE = ['r', 'grade', 'sub_grade', 'installment', 'Tm']
SETS = {'LC가격만': ['r', 'grade', 'sub_grade', 'Tm'], '가격제외': [c for c in F if c not in PRICE], '전체': F}
rows = []
for tag, tr, va, te in [('시간(≤14/15/16)', d[d.y <= 2014], d[d.y == 2015], d[d.y == 2016])]:
    for nm, cols in SETS.items():
        p = fit(tr[cols], tr.bad, va[cols], va.bad, 'bin').predict_proba(te[cols])[:, 1]
        for share in [.2, .3, .5]:
            a = p <= np.quantile(p, share)
            rows.append(dict(분할=tag, 특성=nm, AUC=roc_auc_score(te.bad, p), 승인비율=share, 승인_평균수익=te.RC.values[a].mean(), 승인_부도율=te.bad.values[a].mean()))
pd.DataFrame(rows).round(4).to_csv(f'{OUT}/E_info.csv', index=False); log('E 완료')

# ---------- D: 가상 스냅샷 ----------
def asof(L, C):
    """시점 C까지 관측 가능한 경과개월, 사건."""
    age = ((C.year - L.ym.dt.year) * 12 + C.month - L.ym.dt.month).values
    known = L.res.values & (L.tau.values <= age)
    return age, known
def person_period(L, age, known, cols):
    Q = np.where(known, np.ceil(L.tau.values / 3), np.floor(np.minimum(age, L.Tm.values) / 3)).astype(int)
    keep = Q > 0; idx = np.repeat(np.where(keep)[0], Q[keep])
    q = np.concatenate([np.arange(1, k + 1) for k in Q[keep]])
    X = L[cols].iloc[idx].reset_index(drop=True); X['q'] = q.astype('float32')
    last = np.zeros(len(idx), bool); last[np.cumsum(Q[keep]) - 1] = True
    y = np.where(last & known[idx], L.ev.values[idx], 0)
    return X, y
def ev_score(m, L, cols, rho, rf=.05):
    P_, A, i, T = L.funded_amnt.values.astype(float), L.installment.values.astype(float), L.r.values / 12, L.Tm.values
    g = (1 + rf) ** (1 / 12) - 1; S = np.ones(len(L)); E = np.zeros(len(L)); X = L[cols].copy()
    B = lambda k: np.maximum(P_ * (1 + i) ** k - A * ((1 + i) ** k - 1) / i, 0)
    for q in range(1, 21):
        act = 3 * q <= T
        if not act.any(): break
        X['q'] = np.float32(q); pr = m.predict_proba(X)
        hd, hp = pr[:, 1], pr[:, 2]
        f = lambda k: (1 + g) ** np.maximum(T - k, 0)
        pay = A * (f(3 * q - 2) + f(3 * q - 1) + f(3 * q))
        E += act * S * ((1 - hd) * pay + hp * B(3 * q) * f(3 * q) + hd * rho * B(3 * (q - 1)) * f(3 * q))
        S = S * (1 - hd - hp) * act + S * (~act)
    return np.clip(E / P_, 1e-3, None) ** (12 / T) - 1
HF = F if not SMOKE else F[:40]
rows = []
for Y in [2014, 2015, 2016]:
    C = pd.Timestamp(f'{Y}-01-01'); L = raw[raw.ym < C]; te = d[d.y == Y]
    age, known = asof(L, C); R = L[known]
    log('D', Y, '학습 가능 대출', len(L), '그 시점 종결', len(R), f'종결 부도율 {R.bad.mean():.3f} / 최종 부도율 {L[L.res].bad.mean():.3f}')
    u = np.random.default_rng(Y).random(len(R)) < .85
    sc = {}
    sc['금리 낮은순'] = -te.r.values
    sc['A/B등급 우선'] = -te.grade.cat.codes.values.astype(float) - te.r.values
    sc['시점종결 PD'] = -fit(R[F][u], R.bad[u], R[F][~u], R.bad[~u], 'bin').predict_proba(te[F])[:, 1]
    e = R.RC - .05
    sc['시점종결 기대수익'] = fit(R[F][u], e[u], R[F][~u], e[~u], 'reg').predict(te[F])
    Lf = L[L.res]; uf = np.random.default_rng(Y + 1).random(len(Lf)) < .85; ef = Lf.RC - .05
    sc['미래정보 PD(상한)'] = -fit(Lf[F][uf], Lf.bad[uf], Lf[F][~uf], Lf.bad[~uf], 'bin').predict_proba(te[F])[:, 1]
    sc['미래정보 기대수익(상한)'] = fit(Lf[F][uf], ef[uf], Lf[F][~uf], ef[~uf], 'reg').predict(te[F])
    samp = np.random.default_rng(Y + 2).random(len(L)) < (.6 if not SMOKE else 1)
    X, yy = person_period(L[samp], age[samp], known[samp], HF)
    log('  person-period', X.shape, np.bincount(yy))
    hz = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=60 if SMOKE else 400, learning_rate=.05, num_leaves=63, subsample=.5, subsample_freq=1,
                            colsample_bytree=.6, min_child_samples=500, reg_lambda=5, n_jobs=-1, verbose=-1).fit(X, yy)
    del X
    co = R[R.ev == 1]; bal = (co.funded_amnt - co.total_rec_prncp).astype(float)
    rho = float(co.recoveries.sum() / max(bal.sum(), 1)); log('  회수율', round(rho, 4))
    sc['생존모형 기대수익(Current 활용)'] = ev_score(hz, te, HF, rho)
    for nm, s in sc.items():
        for share in [.1, .2, .3, .5, .7]:
            a = s >= np.quantile(s, 1 - share)
            rows.append(dict(테스트연도=Y, 전략=nm, 승인비율=share, 평균수익=te.RC.values[a].mean(), 부도율=te.bad.values[a].mean(), 평균금리=te.r.values[a].mean(),
                             샤프5=sharpe((te.RC - .05).values, a), 샤프2=sharpe((te.RD - .02).values, a)))
    rows.append(dict(테스트연도=Y, 전략='전부 승인', 승인비율=1.0, 평균수익=te.RC.mean(), 부도율=te.bad.mean(), 평균금리=te.r.mean(),
                     샤프5=sharpe((te.RC - .05).values, np.ones(len(te), bool)), 샤프2=sharpe((te.RD - .02).values, np.ones(len(te), bool))))
    pd.DataFrame(rows).round(4).to_csv(f'{OUT}/D_snapshot.csv', index=False)
log('D 완료')

# ---------- C: 반복 분할 ----------
fC = f'{OUT}/C_repeat.csv'
done = set(pd.read_csv(fC).seed) if os.path.exists(fC) else set()
def metrics(te, a, R, rf):
    e = (te[R] - rf).values; x = np.where(a, e, 0.0)
    z = pd.DataFrame({'x': x, 'ym': te.ym.values, 'y': te.y.values}); z = z[z.y.between(2012, 2016)]
    m = z.groupby('ym').x.mean()
    yrs = (np.where(te.tau.values > 0, te.tau.values, 1) / 12)
    prof = ((te.total_pymnt - te.funded_amnt).values / yrs)
    roe = (np.where(a, prof, te.funded_amnt.values * rf)).sum() / te.funded_amnt.values.sum()
    return dict(승인비율=a.mean(), 샤프=sharpe(e, a), 소르티노=x.mean() / max(np.sqrt((np.minimum(x, 0) ** 2).mean()), 1e-9), 코호트비=m.mean() / m.std(ddof=1),
                승인평균=e[a].mean() if a.any() else np.nan, 승인하위5=np.quantile(e[a], .05) if a.any() else np.nan, 승인부도율=te.bad.values[a].mean() if a.any() else np.nan, 지난ROE식=roe)
for seed in range(NSEEDS):
    if seed in done: continue
    u = np.random.default_rng(1000 + seed).random(len(d))
    tr, va, te = d[u < .6], d[(u >= .6) & (u < .8)], d[u >= .8]
    Xt, Xv, Xe = tr[F], va[F], te[F]
    pdm = fit(Xt, tr.bad, Xv, va.bad, 'bin'); pv, pt = pdm.predict_proba(Xv)[:, 1], pdm.predict_proba(Xe)[:, 1]
    # 2단계: p_ratio_hat (표본 내 vs out-of-fold)
    fold = np.random.default_rng(seed).integers(0, 3, len(tr)); oof = np.zeros(len(tr))
    for k in range(3):
        m = fit(Xt[fold != k], tr.pratio[fold != k], Xv, va.pratio, 'reg'); oof[fold == k] = m.predict(Xt[fold == k])
    pm = fit(Xt, tr.pratio, Xv, va.pratio, 'reg'); ins, hv, he = pm.predict(Xt), pm.predict(Xv), pm.predict(Xe)
    st = {}
    for nm, h in [('2단계(OOF)', oof), ('2단계(표본내)', ins)]:
        m = fit(Xt.assign(ph=h), tr.bad, Xv.assign(ph=hv), va.bad, 'bin')
        st[nm] = (-m.predict_proba(Xv.assign(ph=hv))[:, 1], -m.predict_proba(Xe.assign(ph=he))[:, 1])
    rec = []
    for R, rf in [('RC', .05), ('RD', .02), ('RA', .05)]:
        ev_, et_, etr = (va[R] - rf).values, (te[R] - rf).values, (tr[R] - rf).values
        S = {'PD컷': (-pv, -pt), '금리낮은순': (-va.r.values, -te.r.values), **st}
        if R == 'RA':
            f = lambda p, r: ((1 - p) * (r - rf) - p * rf) / ((1 - p) * (r - rf) ** 2 + p * rf ** 2)
            S['해석식 E/E2'] = (f(pv, va.r.values), f(pt, te.r.values))
        else:
            pf = fit(Xt, (etr > 0).astype(int), Xv, (ev_ > 0).astype(int), 'bin'); S['수익성여부'] = (pf.predict_proba(Xv)[:, 1], pf.predict_proba(Xe)[:, 1])
            mu = fit(Xt, etr, Xv, ev_, 'reg'); m2 = fit(Xt, etr ** 2, Xv, ev_ ** 2, 'reg')
            a1, a2 = mu.predict(Xv), mu.predict(Xe); b1, b2 = np.clip(m2.predict(Xv), 1e-6, None), np.clip(m2.predict(Xe), 1e-6, None)
            S['기대수익컷'] = (a1, a2); S['E/E2컷'] = (a1 / b1, a2 / b2)
        if seed < 5 and R == 'RC':
            sh = np.random.default_rng(seed).permutation(tr.bad.values)
            m = fit(Xt, sh, Xv, va.bad, 'bin', trees=200); S['플라시보(라벨섞음)'] = (-m.predict_proba(Xv)[:, 1], -m.predict_proba(Xe)[:, 1])
        rec.append(dict(seed=seed, R=R, 전략='전부승인', **metrics(te, np.ones(len(te), bool), R, rf)))
        for nm, (sv, s2) in S.items():
            a = s2 >= pick(sv, ev_)
            rec.append(dict(seed=seed, R=R, 전략=nm, **metrics(te, a, R, rf)))
            if nm == 'PD컷':
                rr = np.random.default_rng(seed).random(len(te)) < a.mean()
                rec.append(dict(seed=seed, R=R, 전략='무작위승인(같은비율)', **metrics(te, rr, R, rf)))
    rec = pd.DataFrame(rec); rec['AUC_PD'] = roc_auc_score(te.bad, pt)
    rec.round(5).to_csv(fC, mode='a', header=not os.path.exists(fC), index=False)
    log('C seed', seed, '완료')
log('전체 완료')
