import numpy as np
import pandas as pd
from lc import benchmarks


def frame():
    return pd.DataFrame({'ym': pd.to_datetime(['2014-01-01', '2016-06-01']), 'term_m': [36, 60]})


def test_premium_adds_to_matched_treasury():
    d = frame()
    base = benchmarks.benchmark(d, '만기 국채')
    assert np.allclose(benchmarks.benchmark(d, '만기 국채 +2%p') - base, 0.02)


def test_ordering_of_benchmarks_in_2014():
    d = frame()
    t3 = benchmarks.benchmark(d, '3개월 국채')[0]
    tm = benchmarks.benchmark(d, '만기 국채')[0]
    baa = benchmarks.benchmark(d, '회사채 BAA')[0]
    assert t3 < tm < baa  # 단기 < 중기 국채 < 회사채
