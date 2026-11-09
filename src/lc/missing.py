"""결측 처리 방침. 결측 유형(결측 검증 시트, 같이 비는 묶음 분석 기준)에 따라 다르게 처리한다.
처리 기준값(중앙값, 최댓값)은 학습 데이터에서만 정한다.

유형
- JOINT  구조적(공동신청 전용): 개인 신청이면 해당 없음
- EVENT  사건 없음: 연체·조회 이력이 없어서 '마지막 ~ 이후 개월'이 없음
- NEWBLK 시기(MAR): 나중에 생긴 변수 묶음(약 29.5% 결측)
- MNAR   MNAR 의심: 근속 연수(emp_length)
- 그 밖 1% 이상 결측은 옛 대출 묶음(MAR), 1% 미만은 소량
"""
import numpy as np
import pandas as pd

EVENT = ['mths_since_last_record', 'mths_since_recent_bc_dlq', 'mths_since_last_major_derog',
         'mths_since_recent_revol_delinq', 'mths_since_last_delinq', 'mths_since_recent_inq']
NEWBLK = ['il_util', 'mths_since_rcnt_il', 'all_util', 'total_cu_tl', 'inq_last_12m', 'total_bal_il', 'inq_fi', 'max_bal_bc',
          'open_rv_24m', 'open_rv_12m', 'open_il_24m', 'open_il_12m', 'open_act_il', 'open_acc_6m']
MNAR_CAT = ['emp_length']
SENTINEL = 9999.0


def is_joint(c):
    return c.startswith('sec_app_') or c.endswith('_joint')


def classify(cols, miss_rate):
    """컬럼 -> 유형."""
    out = {}
    for c in cols:
        r = miss_rate.get(c, 0.0)
        if r == 0:
            continue
        if is_joint(c):
            out[c] = 'JOINT'
        elif c in EVENT:
            out[c] = 'EVENT'
        elif c in NEWBLK:
            out[c] = 'NEWBLK'
        elif c in MNAR_CAT:
            out[c] = 'MNAR'
        elif r >= 0.01:
            out[c] = 'OLDBLK'
        else:
            out[c] = 'SMALL'
    return out


class MissingPolicy:
    """strategy
    - native        : 아무것도 하지 않음(트리 모델 기본)
    - median        : 숫자 결측을 모두 학습 중앙값으로
    - median_flags  : 중앙값 + 결측 표시(같은 패턴의 표시는 하나만)
    - typed_tree    : EVENT는 9999('아주 오래전'), emp_length 결측은 'Unknown' 범주, 나머지는 그대로
    - typed_linear  : EVENT는 학습 최댓값 + 표시, JOINT는 평균, 묶음은 중앙값 + 표시, emp_length는 'Unknown', 소량은 중앙값
    """

    def __init__(self, strategy):
        self.strategy = strategy

    def fit(self, X):
        num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].dtype != bool]
        rate = X.isna().mean()
        self.types = classify(list(X.columns), rate.to_dict())
        self.num = num
        self.med = X[num].median()
        self.mean = X[num].mean()
        self.max = X[num].max()
        # 결측 표시: 1% 이상 결측인 숫자 컬럼(공동신청 제외) 중 서로 다른 패턴만
        cand = [c for c in num if rate[c] >= 0.01 and not is_joint(c)]
        self.flags, seen = [], []
        sample = X[cand].isna().iloc[:: max(1, len(X) // 50000)]
        for c in cand:
            v = sample[c].to_numpy()
            if not any(np.array_equal(v, s) for s in seen):
                seen.append(v)
                self.flags.append(c)
        return self

    def transform(self, X):
        X = X.copy()
        s = self.strategy
        if s == 'native':
            return X
        if s == 'typed_tree':
            for c, t in self.types.items():
                if t == 'EVENT':
                    X[c] = X[c].fillna(SENTINEL)
                elif t == 'MNAR':
                    X[c] = X[c].cat.add_categories('Unknown').fillna('Unknown') if isinstance(X[c].dtype, pd.CategoricalDtype) else X[c].fillna('Unknown')
            return X
        flags = {}
        if s in ('median_flags', 'typed_linear'):
            flags = {f'na_{c}': X[c].isna().astype('float32') for c in self.flags}
        if s in ('median', 'median_flags'):
            X[self.num] = X[self.num].fillna(self.med)
        elif s == 'typed_linear':
            for c in self.num:
                t = self.types.get(c)
                if t == 'EVENT':
                    X[c] = X[c].fillna(self.max[c])
                elif t == 'JOINT':
                    X[c] = X[c].fillna(self.mean[c])
                else:
                    X[c] = X[c].fillna(self.med[c])
        for c, t in self.types.items():
            if t == 'MNAR' and s == 'typed_linear':
                X[c] = X[c].astype('string').fillna('Unknown')
        if flags:
            X = pd.concat([X, pd.DataFrame(flags, index=X.index)], axis=1)
        return X
