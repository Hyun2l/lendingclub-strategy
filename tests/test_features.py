import numpy as np
import pandas as pd
from lc import features


def test_engineered_values():
    d = pd.DataFrame({'annual_inc': [60000.0, 0.0], 'installment': [500.0, 100.0], 'loan_amnt': [12000.0, 5000.0],
                      'ym': pd.to_datetime(['2015-06-01', '2015-06-01']), 'emp_length': ['10+ years', '< 1 year'],
                      'fico_range_low': [700.0, 660.0], 'fico_range_high': [704.0, 664.0]})
    e = features.engineered(d, pd.Series(['Jan-2010', 'Jun-2014']))
    assert abs(e['pti'][0] - 0.1) < 1e-6 and abs(e['lti'][0] - 0.2) < 1e-6
    assert np.isnan(e['pti'][1])  # 소득 0은 결측
    assert e['credit_hist_m'].tolist() == [65.0, 12.0]
    assert e['emp_years'].tolist() == [10.0, 0.5]
    assert e['fico_avg'][0] == 702.0


def test_zip_encoding_out_of_fold_does_not_use_own_label():
    # 한 우편번호에 한 건뿐이면, 교차 인코딩 값은 자기 결과를 쓰지 않아 전체 평균에 가깝다
    z = ['100xx', '200xx', '200xx', '200xx']
    y = [1, 0, 0, 0]
    oof, _ = features.zip3_encoding(z, y, ['100xx'], prior_n=1, folds=4, seed=1)
    assert oof[0] < 0.5
