"""Runnable examples for the code cheat sheet. No project data needed.

Run all:  python docs/notes/code/snippets.py
Run one:  python docs/notes/code/snippets.py irr
          (basics, mask, groupby, asof, irr, cut, sharpe, boot, onese, pareto, lgbm)
Needs numpy, pandas, scipy (lgbm demo also needs lightgbm, scikit-learn).
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ---------------------------------------------------------------- 1. DataFrame basics
def basics():
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'ym': pd.to_datetime(['2013-01', '2013-01', '2013-01', '2013-02', '2013-02', '2013-02']),
        'grade': ['A', 'B', 'C', 'A', 'B', 'C'],
        'amt': [1000, 2000, 1500, 1000, 3000, 500],
        'r': [0.07, 0.11, 0.15, 0.07, 0.12, 0.16],
        'status': ['Fully Paid', 'Charged Off', 'Fully Paid', 'Fully Paid', 'Fully Paid', 'Charged Off'],
    })
    print(df)
    print('rows:', len(df), '/ columns:', list(df.columns))
    print('mean rate:', df['r'].mean())
    df['bad'] = (df['status'] == 'Charged Off').astype(int)   # True/False -> 1/0
    print(df[['id', 'status', 'bad']])
    return df


# ---------------------------------------------------------------- 2. Boolean mask (pick rows by condition)
def mask():
    df = basics()
    m = (df['grade'] == 'A') | (df['r'] > 0.14)          # | or, & and, ~ not
    print('mask:', m.tolist())
    print(df[m][['id', 'grade', 'r']])
    print('np.where:', np.where(df['bad'] == 1, 0.0, df['r']))  # 0 if default else rate


# ---------------------------------------------------------------- 3. groupby
def groupby():
    df = basics()
    print(df.groupby('ym')['amt'].sum())                 # sum per month
    print(df.groupby('grade')['bad'].mean())             # default rate per grade
    idx = df.groupby('ym').indices                       # {group value: array of row numbers}
    for k, ix in idx.items():
        print(pd.Timestamp(k).strftime('%Y-%m'), '-> rows', ix)


# ---------------------------------------------------------------- 4. As-of training: only what was known then
def asof():
    d = pd.DataFrame({
        'ym': pd.to_datetime(['2014-01', '2014-06', '2015-03', '2015-09']),
        'tau': [12, 6, 3, 2],          # months from issue to last payment
        'bad': [0, 1, 0, 1],
        'resolved': [True, True, True, True],
    })
    cutoff = pd.Timestamp('2016-01-01')
    age = (cutoff.year - d['ym'].dt.year) * 12 + cutoff.month - d['ym'].dt.month   # months from issue to cutoff
    settle = d['tau'] + np.where(d['bad'] == 1, 5, 0)                              # defaults settle 5 months later
    d['age'], d['settle'] = age, settle
    d['known'] = (age > 0) & d['resolved'] & (settle <= age)
    print(d)


# ---------------------------------------------------------------- 5. Cash flows and IRR
def annuity(P, rate, T):
    """Level monthly payment."""
    g = rate / 12
    return P * g / (1 - (1 + g) ** -T)


def irr(cf):
    """Monthly cash flows -> annual IRR. cf[0] is the investment (negative)."""
    f = lambda m: (cf / (1 + m) ** np.arange(len(cf))).sum()   # sum of present values
    m = brentq(f, -0.95, 1.0)                                  # find m with f(m) = 0
    return (1 + m) ** 12 - 1


def irr_demo():
    P, r, T = 1000, 0.12, 36
    inst = annuity(P, r, T)
    print(f'monthly payment {inst:.2f}')
    cf = np.r_[-P, np.full(T, inst)]                # fully paid
    print(f'fully paid: IRR {irr(cf):.4f}')
    cf_fee = np.r_[-P, np.full(T, inst) * 0.99]     # 1% fee
    print(f'1% fee: IRR {irr(cf_fee):.4f}')
    cf_def = np.zeros(T + 1)                        # 10 payments, default, recover 100 in month 15
    cf_def[0] = -P
    cf_def[1:11] = inst
    cf_def[15] = 100
    print(f'default after 10: IRR {irr(cf_def):.4f}')
    pool = np.r_[-2 * P, cf[1:] + cf_def[1:]]       # two loans as one pool
    pool[0] = -2 * P
    print(f'pool of two: IRR {irr(pool):.4f}')


# ---------------------------------------------------------------- 6. Top q within each issue month (with ties)
def cut(score, groups, q):
    a = np.zeros(len(score), bool)
    for _, ix in pd.Series(groups).groupby(groups).indices.items():
        s = score[ix]
        a[ix] = s >= np.quantile(s, 1 - q)
    return a


def cut_demo():
    ym = np.array(['01', '01', '01', '01', '02', '02', '02', '02'])
    score = np.array([0.9, 0.5, 0.5, 0.1, 0.3, 0.8, 0.6, 0.2])
    print('top 50%:', cut(score, ym, 0.5).astype(int))
    tie = np.array([1, 1, 1, 1, 2, 2, 2, 2], float)   # all tied?
    print('ties as is:', cut(tie, ym, 0.5).astype(int), '-> approves all, not 50%')
    jit = np.random.default_rng(11).random(len(tie)) * 1e-6
    print('ties + tiny noise:', cut(tie + jit, ym, 0.5).astype(int))


# ---------------------------------------------------------------- 7. Portfolio excess return and Sharpe
sharpe = lambda x: x.mean() / x.std(ddof=1)


def sharpe_demo():
    rng = np.random.default_rng(0)
    pool = rng.normal(0.04, 0.02, 48)        # 48 months, pool IRR - treasury
    for w in [0.3, 0.8]:
        port = w * pool                      # rejected money sits in treasury (excess 0)
        print(f'share {w}: mean {port.mean():.4f}, std {port.std(ddof=1):.4f}, Sharpe {sharpe(port):.2f}')
    print('-> constant share leaves Sharpe unchanged')


# ---------------------------------------------------------------- 8. Moving block bootstrap
def block_index(n, block, B, seed=2026):
    st = np.random.default_rng(seed).integers(0, n - block + 1, size=(B, int(np.ceil(n / block))))
    return np.concatenate([np.arange(block)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n]


def boot_demo():
    print(block_index(n=12, block=3, B=3))
    x = np.random.default_rng(1).normal(0.04, 0.02, 48)
    BIX = block_index(48, 6, 2000)
    bs = x[BIX].mean(axis=1)                 # 2000 resampled means
    print(f'mean {x.mean():.4f}, 90% interval {np.quantile(bs, 0.05):.4f} to {np.quantile(bs, 0.95):.4f}')


# ---------------------------------------------------------------- 9. Pick with the one-SE rule
def onese_demo():
    C = pd.DataFrame({'rule': ['A', 'A', 'A', 'B', 'B', 'B'], 'q': [0.1, 0.5, 0.9] * 2,
                      'Sharpe': [3.8, 3.1, 2.9, 2.6, 3.0, 2.9], 'SE': [0.8, 0.8, 0.7, 0.5, 0.7, 0.7]})
    best = C.loc[C['Sharpe'].idxmax()]
    ok = C[C['Sharpe'] >= best['Sharpe'] - best['SE']]
    pick = ok.sort_values(['q', 'Sharpe'], ascending=[False, False]).iloc[0]
    print(C)
    print(f"best {best['rule']} {best['q']}: {best['Sharpe']} - SE {best['SE']} = bar {best['Sharpe'] - best['SE']:.1f}")
    print('pass:', ok[['rule', 'q']].values.tolist())
    print('pick:', pick['rule'], pick['q'])


# ---------------------------------------------------------------- 10. Pareto (dominance)
def dominated(mean, std):
    """True if another point has mean >= and std <=, strictly better in one."""
    m, s = np.asarray(mean), np.asarray(std)
    return np.array([bool(((m >= m[i]) & (s <= s[i]) & ((m > m[i]) | (s < s[i]))).any()) for i in range(len(m))])


def pareto_demo():
    names = ['ER', 'lowPD', 'simple', 'random']
    mean = [1.88, 1.66, 1.84, 1.46]
    std = [0.65, 0.50, 0.48, 0.54]
    for n_, d_ in zip(names, dominated(mean, std)):
        print(n_, 'dominated' if d_ else 'frontier')


# ---------------------------------------------------------------- 11. Default probability with LightGBM
def lgbm_demo():
    import lightgbm as lgb
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    X, y = make_classification(n_samples=5000, n_features=10, weights=[0.85], random_state=0)
    Xtr, ytr, Xte, yte = X[:4000], y[:4000], X[4000:], y[4000:]
    ps = []
    for seed in [0, 1, 2]:
        u = np.random.default_rng(seed).random(len(Xtr)) < 0.15          # 15% held out for early stopping
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                               random_state=seed, verbose=-1)
        m.fit(Xtr[~u], ytr[~u], eval_set=[(Xtr[u], ytr[u])], callbacks=[lgb.early_stopping(50, verbose=False)])
        ps.append(m.predict_proba(Xte)[:, 1])                             # probability of class 1 (default)
        print(f'seed {seed}: {m.best_iteration_} trees, AUC {roc_auc_score(yte, ps[-1]):.3f}')
    p = np.mean(ps, axis=0)
    print(f'3-seed mean AUC {roc_auc_score(yte, p):.3f}, mean p {p.mean():.3f} vs actual {yte.mean():.3f}')


DEMOS = {'basics': basics, 'mask': mask, 'groupby': groupby, 'asof': asof, 'irr': irr_demo, 'cut': cut_demo,
         'sharpe': sharpe_demo, 'boot': boot_demo, 'onese': onese_demo, 'pareto': pareto_demo, 'lgbm': lgbm_demo}

if __name__ == '__main__':
    for k in (sys.argv[1:] or DEMOS):
        print(f'\n===== {k} =====')
        DEMOS[k]()
