import numpy as np
import pandas as pd
from lc import missing


def frame():
    return pd.DataFrame({'mths_since_last_delinq': [5.0, np.nan, 20.0, np.nan],
                         'sec_app_fico_range_low': [np.nan, 700.0, np.nan, np.nan],
                         'il_util': [np.nan, 50.0, 60.0, np.nan],
                         'dti': [10.0, 20.0, 30.0, 40.0],
                         'emp_length': pd.Categorical(['1 year', None, '10+ years', '1 year'])})


def test_classify_types():
    X = frame()
    t = missing.classify(list(X.columns), X.isna().mean().to_dict())
    assert t == {'mths_since_last_delinq': 'EVENT', 'sec_app_fico_range_low': 'JOINT', 'il_util': 'NEWBLK', 'emp_length': 'MNAR'}


def test_typed_linear_fills_from_training_only():
    X = frame()
    p = missing.MissingPolicy('typed_linear').fit(X)
    new = pd.DataFrame({'mths_since_last_delinq': [np.nan], 'sec_app_fico_range_low': [np.nan], 'il_util': [np.nan],
                        'dti': [np.nan], 'emp_length': pd.Categorical([None])})
    out = p.transform(new)
    assert out['mths_since_last_delinq'].iloc[0] == 20.0        # 학습 최댓값('아주 오래전')
    assert out['sec_app_fico_range_low'].iloc[0] == 700.0       # 학습 평균
    assert out['il_util'].iloc[0] == 55.0                        # 학습 중앙값
    assert out['emp_length'].iloc[0] == 'Unknown'
    assert out['na_mths_since_last_delinq'].iloc[0] == 1.0


def test_typed_tree_sentinel_and_unknown():
    X = frame()
    out = missing.MissingPolicy('typed_tree').fit(X).transform(X)
    assert out['mths_since_last_delinq'].tolist() == [5.0, 9999.0, 20.0, 9999.0]
    assert out['emp_length'].isna().sum() == 0 and 'Unknown' in list(out['emp_length'])
    assert out['il_util'].isna().sum() == 2                      # 나머지는 그대로
