"""데이터 불러오기, 결과 상태 분류, 누수 제거, 특성 정리."""
import numpy as np
import pandas as pd

from .config import TRAIN_PARQUET

# 대출 승인 이후에야 알 수 있는 정보(누수). 모델 입력에 절대 넣지 않는다.
LEAK = [
    'loan_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
    'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',
    'debt_settlement_flag', 'pymnt_plan', 'deferral_term', 'payment_plan_start_date',
    'orig_projected_additional_accrued_interest',
]
LEAK_PREFIX = ('hardship_', 'settlement_')
# 식별자, 자유 입력 텍스트, 상수 컬럼. 문자열 원본(int_rate, term, issue_d)은 파생 숫자로 대체한다.
NON_FEATURE = ['id', 'url', 'emp_title', 'title', 'zip_code', 'policy_code', 'int_rate', 'term', 'issue_d']
# parse()가 만드는 결과·시점 컬럼. 이 중 r, term_m은 승인 시점에 알 수 있으므로 특성으로 쓴다.
DERIVED_OUTCOME = ['ym', 'issue_year', 'tau', 'outcome', 'resolved', 'bad', 'policy_fail']

OUTCOMES = ('paid', 'default', 'late', 'current')


def load(path=TRAIN_PARQUET, columns=None):
    """원본 캐시를 읽고 빈 행(loan_status 결측)을 제거한다."""
    df = pd.read_parquet(path, columns=columns)
    return df[df['loan_status'].notna()].reset_index(drop=True)


def classify(status):
    """loan_status -> paid / default / late / current."""
    s = status.astype(str)
    return pd.Series(np.select(
        [s.str.contains('Fully Paid'), s.str.contains('Charged Off') | s.eq('Default'),
         s.str.startswith('Late') | s.eq('In Grace Period')],
        ['paid', 'default', 'late'], 'current'), index=status.index)


def months_between(start, end):
    return (end.dt.year - start.dt.year) * 12 + end.dt.month - start.dt.month


def parse(df):
    """문자열 컬럼을 숫자로 바꾸고 결과·시점 컬럼을 추가한다(원본은 바꾸지 않음)."""
    d = df.copy()
    d['ym'] = pd.to_datetime(d['issue_d'], format='%b-%Y')
    d['issue_year'] = d['ym'].dt.year
    d['r'] = pd.to_numeric(d['int_rate'].astype('string').str.strip().str.rstrip('%')) / 100
    d['term_m'] = d['term'].astype('string').str.extract(r'(\d+)')[0].astype(int)
    if 'revol_util' in d:
        d['revol_util'] = pd.to_numeric(d['revol_util'].astype('string').str.rstrip('%'), errors='coerce')
    for c in ['earliest_cr_line', 'sec_app_earliest_cr_line']:
        if c in d:
            d[c] = pd.to_datetime(d[c], format='%b-%Y', errors='coerce').dt.year
    if 'last_pymnt_d' in d:
        lp = pd.to_datetime(d['last_pymnt_d'], format='%b-%Y', errors='coerce')
        # 마지막 납부까지 경과 개월. 납부 기록이 없으면 1로 둔다.
        d['tau'] = months_between(d['ym'], lp).fillna(1).clip(lower=1).astype(int)
    d['outcome'] = classify(d['loan_status'])
    d['policy_fail'] = d['loan_status'].astype(str).str.startswith('Does not meet')
    d['resolved'] = d['outcome'].isin(['paid', 'default'])
    d['bad'] = (d['outcome'] == 'default').astype('int8')
    return d


def resolved_sample(d, include_policy_fail=False):
    """종결 대출 표본. 기본값은 신용정책 미충족 대출(2007~2010, 1,683건)을 뺀다."""
    m = d['resolved'] & (include_policy_fail | ~d['policy_fail'])
    return d[m]


def is_leak(col):
    return col in LEAK or col.startswith(LEAK_PREFIX)


def feature_columns(d):
    """승인 시점에 알 수 있는 특성만 반환한다."""
    return [c for c in d.columns if not is_leak(c) and c not in NON_FEATURE and c not in DERIVED_OUTCOME]


def model_frame(d, cols):
    """숫자는 float32, 문자열은 category로 바꾼 모델 입력."""
    X = d[cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]) and X[c].dtype != bool:
            X[c] = X[c].astype('float32')
        else:
            X[c] = X[c].astype('category')
    return X
