import pandas as pd
from lc import data


def test_classify_status():
    s = pd.Series(['Fully Paid', 'Charged Off', 'Default', 'Current', 'Late (16-30 days)', 'In Grace Period',
                   'Issued', 'Does not meet the credit policy. Status:Fully Paid',
                   'Does not meet the credit policy. Status:Charged Off'])
    assert data.classify(s).tolist() == ['paid', 'default', 'default', 'current', 'late', 'late',
                                         'current', 'paid', 'default']


def test_leak_columns_never_features():
    cols = ['loan_amnt', 'grade', 'r', 'term_m', 'total_pymnt', 'recoveries', 'hardship_flag',
            'settlement_amount', 'last_pymnt_d', 'loan_status', 'id', 'url', 'bad', 'tau']
    d = pd.DataFrame(columns=cols)
    assert data.feature_columns(d) == ['loan_amnt', 'grade', 'r', 'term_m']
