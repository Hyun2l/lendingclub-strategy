"""비교 기준(거절한 돈을 대신 넣을 곳)의 연 수익률.

- 만기 국채: 발행월·만기에 맞춘 미 국채(GS3, GS5). 과제가 정한 '국채'의 기본 해석.
- 3개월 국채: 교과서적 무위험(TB3MS). 신용·금리 위험이 모두 없다.
- 회사채 BAA: 같은 '신용 투자'의 대안(Moody's Baa). 약속 수익률이라 실제보다 약간 높다.
- 만기 국채 + 프리미엄: LC 대출은 중도 매도가 어려워 더 받아야 한다는 유동성 보상.
하이일드 지수는 FRED에서 2023년 이후만 제공되어 쓰지 않는다.
"""
import pandas as pd

from .config import DATA
from . import returns


def _series(code):
    x = pd.read_csv(DATA / 'external' / f'{code}.csv')
    x.columns = ['ym', 'v']
    x['ym'] = pd.to_datetime(x['ym'])
    x['v'] = pd.to_numeric(x['v'], errors='coerce') / 100
    return x.set_index('ym')['v']


def benchmark(d, name):
    """d(ym, term_m 포함)에 맞춘 비교 기준 수익률 배열."""
    if name == '만기 국채':
        return returns.attach_treasury(d)['tb'].to_numpy()
    if name.startswith('만기 국채 +'):
        add = float(name.split('+')[1].rstrip('%p')) / 100
        return returns.attach_treasury(d)['tb'].to_numpy() + add
    code = {'3개월 국채': 'TB3MS', '회사채 BAA': 'BAA'}[name]
    s = _series(code)
    v = d['ym'].map(s).to_numpy()
    assert pd.notna(v).all(), f'{code} 값이 없는 발행월이 있다'
    return v


NAMES = ['3개월 국채', '만기 국채', '만기 국채 +1%p', '만기 국채 +2%p', '회사채 BAA', '만기 국채 +3%p']
