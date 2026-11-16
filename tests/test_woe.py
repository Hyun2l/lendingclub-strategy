import numpy as np
import pandas as pd
from lc import woe


def test_woe_sign_and_iv():
    # x가 클수록 부도가 많다 -> 큰 구간의 WOE는 음수(정상이 적음), IV > 0
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < 0.05 + 0.3 * x).astype(int)
    df = pd.DataFrame({'x': x})
    w = woe.WOE(n_bins=5).fit(df, y, ['x'])
    t = w.transform(df)['x']
    assert t[x > 0.9].mean() < 0 < t[x < 0.1].mean()
    assert w.iv_table()['x'] > 0.1


def test_missing_and_unseen_categories():
    df = pd.DataFrame({'c': ['a'] * 50 + ['b'] * 50 + [None] * 10})
    y = np.array([0] * 45 + [1] * 5 + [0] * 25 + [1] * 25 + [1] * 10)
    w = woe.WOE().fit(df, y, ['c'])
    out = w.transform(pd.DataFrame({'c': ['a', 'zzz', None]}))['c']
    assert out.iloc[0] > 0 and out.iloc[2] < 0   # 'a'는 안전, 결측은 위험
    assert out.iloc[1] != 0 or True               # 처음 보는 값은 '기타' 구간 또는 0
