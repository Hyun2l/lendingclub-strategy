"""모델 없이 만드는 기준선 전략. 모든 새 전략은 이 표와 비교한다."""
import numpy as np


def within_year_bottom(d, col, q):
    """발행연도마다 col이 낮은 하위 q 비율을 승인."""
    a = np.zeros(len(d), bool)
    v = d[col].to_numpy()
    for _, idx in d.groupby('issue_year').indices.items():
        a[idx] = v[idx] <= np.quantile(v[idx], q)
    return a


def baselines(d, seed=0):
    rng = np.random.default_rng(seed)
    return {
        '전부 승인': np.ones(len(d), bool),
        '전부 국채': np.zeros(len(d), bool),
        '무작위 30%': rng.random(len(d)) < 0.3,
        'A·B등급만': d['grade'].isin(['A', 'B']).to_numpy(),
        '금리 낮은 30%': within_year_bottom(d, 'r', 0.3),
        '금리 높은 30%': ~within_year_bottom(d, 'r', 0.7),
    }
