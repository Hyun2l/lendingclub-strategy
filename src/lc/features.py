"""피처 엔지니어링 후보. 모두 대출 신청 시점에 알 수 있는 값으로만 만든다.

- 상환 부담: 연 상환액 / 연소득(pti), 대출액 / 연소득(lti)
- 신용 이력 개월 수: 발행월 - 첫 신용 개설월 (기존 특성은 개설 '연도'만 담음)
- 근속 연수 숫자화: emp_length 문자열 -> 숫자
- FICO 평균: fico_range_low/high 평균 (기존 특성과 거의 같아 '효과 없음' 대조군)
- 우편번호 앞 3자리 부도율: 그 시점까지 확정된 대출로만 계산, 학습 행은 교차(out-of-fold) 인코딩
- 무작위 노이즈: 아무 정보도 없는 플라시보
"""
import numpy as np
import pandas as pd

EMP = {'< 1 year': 0.5, '1 year': 1, '10+ years': 10}


def engineered(d, earliest_cr_line_raw):
    """d: data.parse 결과, earliest_cr_line_raw: 원본 문자열 컬럼(같은 index)."""
    inc = d['annual_inc'].astype(float).where(d['annual_inc'] > 0)
    out = pd.DataFrame(index=d.index)
    out['pti'] = (d['installment'].astype(float) * 12 / inc).clip(upper=2)
    out['lti'] = (d['loan_amnt'].astype(float) / inc).clip(upper=5)
    ec = pd.to_datetime(earliest_cr_line_raw, format='%b-%Y', errors='coerce')
    out['credit_hist_m'] = ((d['ym'].dt.year - ec.dt.year) * 12 + d['ym'].dt.month - ec.dt.month).astype(float)
    s = d['emp_length'].astype('string').str.strip()
    out['emp_years'] = s.map(EMP).astype(float).fillna(pd.to_numeric(s.str.extract(r'(\d+)')[0], errors='coerce'))
    out['fico_avg'] = (d['fico_range_low'].astype(float) + d['fico_range_high'].astype(float)) / 2
    out['noise'] = np.random.default_rng(12345).normal(size=len(d))
    return out.astype('float32')


def zip3_encoding(zip_train, y_train, zip_apply, prior_n=200, folds=5, seed=0):
    """우편번호 앞 3자리별 부도율(평활). 학습 행은 교차 인코딩, 적용 행은 학습 전체로 인코딩."""
    zt = pd.Series(zip_train).astype(str).str[:3].to_numpy()
    za = pd.Series(zip_apply).astype(str).str[:3].to_numpy()
    y = np.asarray(y_train, float)
    g = y.mean()

    def enc(z_fit, y_fit, z_use):
        t = pd.DataFrame({'z': z_fit, 'y': y_fit}).groupby('z')['y'].agg(['sum', 'count'])
        m = (t['sum'] + prior_n * g) / (t['count'] + prior_n)
        return pd.Series(z_use).map(m).fillna(g).to_numpy()

    k = np.random.default_rng(seed).integers(0, folds, len(zt))
    oof = np.zeros(len(zt))
    for f in range(folds):
        oof[k == f] = enc(zt[k != f], y[k != f], zt[k == f])
    return oof.astype('float32'), enc(zt, y, za).astype('float32')
