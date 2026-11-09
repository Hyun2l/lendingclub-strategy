"""WOE(Weight of Evidence) 변환과 IV(Information Value). 신용평가 스코어카드의 표준 도구.

WOE_b = ln( (정상_b / 정상 전체) / (부도_b / 부도 전체) ),  IV = Σ_b (정상_b/정상 - 부도_b/부도) x WOE_b
숫자는 분위 구간 + 결측 구간, 범주는 값별(드문 값은 '기타')로 나눈다. 구간 기준은 학습 데이터에서만 정한다.
"""
import numpy as np
import pandas as pd

MISSING = '__결측__'
OTHER = '__기타__'


class WOE:
    def __init__(self, n_bins=10, min_share=0.005, smooth=0.5):
        self.n_bins, self.min_share, self.smooth = n_bins, min_share, smooth
        self.spec = {}

    def _bin(self, col, x):
        kind, rule = self.spec[col]['kind'], self.spec[col]['rule']
        if kind == 'num':
            b = pd.Series(np.searchsorted(rule, x.to_numpy(float), side='right'), index=x.index).astype(str)
            return b.where(x.notna(), MISSING)
        s = x.astype('string').fillna(MISSING)
        return s.where(s.isin(rule) | (s == MISSING), OTHER).astype(str)

    def fit(self, df, y, cols):
        y = np.asarray(y)
        for c in cols:
            x = df[c]
            if pd.api.types.is_numeric_dtype(x) and x.dtype != bool:
                q = np.unique(np.nanquantile(x.to_numpy(float), np.linspace(0, 1, self.n_bins + 1)[1:-1])) if x.notna().any() else np.array([])
                self.spec[c] = {'kind': 'num', 'rule': q}
            else:
                share = x.astype('string').value_counts(normalize=True)
                self.spec[c] = {'kind': 'cat', 'rule': list(share[share >= self.min_share].index)}
            b = self._bin(c, x)
            t = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0) + self.smooth
            g, d = t[0] / t[0].sum(), t[1] / t[1].sum()
            w = np.log(g / d)
            self.spec[c]['woe'] = w.to_dict()
            self.spec[c]['iv'] = float(((g - d) * w).sum())
        return self

    def transform(self, df, cols=None):
        cols = cols or list(self.spec)
        out = {}
        for c in cols:
            b = self._bin(c, df[c])
            out[c] = b.map(self.spec[c]['woe']).fillna(0.0).astype('float32')  # 처음 보는 구간 = 정보 없음
        return pd.DataFrame(out, index=df.index)

    def iv_table(self):
        return pd.Series({c: s['iv'] for c, s in self.spec.items()}).sort_values(ascending=False)


def iv_strength(iv):
    return '쓸모없음' if iv < 0.02 else '약함' if iv < 0.1 else '중간' if iv < 0.3 else '강함' if iv < 0.5 else '의심(너무 강함)'
