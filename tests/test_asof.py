import pandas as pd
from lc import asof


def frame():
    return pd.DataFrame({'ym': pd.to_datetime(['2015-01-01', '2015-01-01', '2015-06-01']),
                         'issue_year': [2015, 2015, 2015], 'tau': [10, 10, 3],
                         'bad': [0, 1, 0], 'resolved': [True, True, False]})


def test_paid_known_after_last_payment():
    d = frame()
    assert asof.known_at(d, '2015-11-01').tolist() == [True, False, False]
    assert asof.known_at(d, '2015-10-01').tolist() == [False, False, False]


def test_default_known_after_charge_off_lag():
    d = frame()
    assert asof.known_at(d, '2016-04-01').tolist() == [True, True, False]


def test_training_set_has_no_future_loans():
    d = frame()
    tr = asof.training_set(d, '2015-03-01')
    assert (tr['ym'] < pd.Timestamp('2015-03-01')).all()
