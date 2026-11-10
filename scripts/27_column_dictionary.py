"""27단계: 팀 컬럼 사전 엑셀을 지금까지의 결과(1~26단계)로 갱신한다.

사용: python scripts/27_column_dictionary.py [원본 컬럼사전.xlsx]
출력: docs/컬럼사전_최신.xlsx (원본 시트와 열은 그대로 두고, 오른쪽에 확인된 사실 열을 더한다)

숫자는 모두 train 데이터에서 이 스크립트나 앞 단계 결과 파일로 계산한 값이다.
"""
import sys
from copy import copy
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from lc import data, asof, missing
from lc.config import ROOT, RESULTS

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Downloads' / 'lending_club_컬럼사전.xlsx'
OUT = ROOT / 'docs' / '컬럼사전_최신.xlsx'
YEARS = range(2007, 2021)

raw = data.load()
d = data.parse(raw)
feats = set(data.feature_columns(d[~d['policy_fail']]))
res = (d['resolved'] & ~d['policy_fail']).to_numpy()
bad = d['bad'].to_numpy()
yr = d['issue_year'].to_numpy()
dd = d[~d['policy_fail']]
avail = {Y: raw.loc[asof.training_set(dd, f'{Y}-01-01').index].notna().mean() for Y in (2013, 2016)}
mtype = missing.classify(list(raw.columns), raw.isna().mean().to_dict())
TYPE_KO = {'JOINT': '구조적(공동신청 전용)', 'EVENT': '사건 없음(그런 일이 없었음)', 'NEWBLK': '시기(2015년 말 이후 생긴 새 변수)',
           'MNAR': '값 자체와 관련 의심(MNAR)', 'OLDBLK': '시기(옛 대출에 없던 변수)', 'SMALL': '소량(1% 미만)'}
LINEAR = {'JOINT': '가용성 규칙으로 대부분 제외', 'EVENT': '가용성 규칙 → 1~99% 자르기 → 중앙값', 'NEWBLK': '학습 시점 가용률 50% 미만이면 제외, 아니면 중앙값',
          'MNAR': "'모름' 범주로 따로 둠", 'OLDBLK': '가용성 규칙 → 1~99% 자르기 → 중앙값', 'SMALL': '중앙값'}
RETURN_USE = {'loan_status', 'total_pymnt', 'last_pymnt_d', 'installment', 'funded_amnt', 'issue_d', 'int_rate', 'term'}
RENAMED = {'int_rate': 'r', 'term': 'term_m'}

iv = pd.read_csv(RESULTS / 'eda' / 'T_iv.csv', index_col=0)
imp = pd.read_csv(RESULTS / '17_feature_importance.csv').reset_index(drop=True)
imp['순위'] = np.arange(1, len(imp) + 1)
imp = imp.set_index('피처')
wg = pd.read_csv(RESULTS / '12_within_grade_signal.csv', index_col=0)
pairs = pd.read_csv(RESULTS / 'eda' / 'T_high_corr_pairs.csv', header=0)
pairs.columns = ['a', 'b', 'rho']
partner = {}
for a, b, rho in pairs.itertuples(index=False):
    partner.setdefault(a, []).append(f'{b}({rho:.2f})')
    partner.setdefault(b, []).append(f'{a}({rho:.2f})')

NOTE = {
    'annual_inc': '혼자서는 약하지만(IV 0.03) 같은 등급 안에서 가장 강한 변수. 소득 상위 + DTI 하위 20%가 같은 세부등급에서 IRR 약 0.9%p 높음(19단계). 효과는 해마다 줄어듦',
    'dti': '등급이 덜 반영한 상환 능력 정보(19단계)',
    'emp_length': '결측이 무직·미기재와 관련 의심. 선형 모델에서는 결측을 별도 범주로',
    'sub_grade': 'IV 1위. LC 등급끼리의 가격은 대체로 공정했고, 모델의 추가 수익은 같은 등급 안에서 나옴',
    'grade': '성과 귀속(Brinson)의 묶음 기준',
    'int_rate': '숫자 r로 바꿔 입력. 기대수익 점수 (1-p)r - p·LGD - 국채 에 직접 쓰임',
    'term': '36/60개월. 주 평가는 36개월, 비교 국채 만기를 맞추는 데 씀',
    'installment': '모델 입력이면서 현금흐름 복원(정상 납입 개월 계산)에도 씀',
    'total_pymnt': '받은 총액. 수익률(IRR)과 손실률(LGD) 계산에만 사용',
    'last_pymnt_d': '마지막 납부 달. 현금흐름 시점과 시점 학습의 확정 시점(부도는 +5개월)에 사용',
    'loan_status': '정답(부도 여부)과 종결 여부를 만듦. 진행 중 대출은 평가에서 제외',
    'issue_d': '발행월. 시점 학습, 코호트(발행월 묶음), 국채 매칭의 기준',
    'last_fico_range_high': '대출 후 갱신된 점수라 부도가 나면 떨어짐. 넣으면 AUC가 0.9999까지 오르는 대표적 누수',
    'zip_code': '앞 3자리 부도율로 만든 피처는 효과가 시드 차이 안이라 기각(13단계)',
    'acc_open_past_24mths': '등급 안 정보가 남아 있는 변수(12단계)',
}

rows = []
for c in raw.columns:
    m = raw[c].isna().to_numpy()
    if c in data.LEAK or c.startswith(data.LEAK_PREFIX):
        use = '누수로 제외(수익·정답 계산에만 사용)' if c in RETURN_USE else '누수로 제외'
    elif c in RENAMED:
        use = f'입력({RENAMED[c]}로 변환)'
    elif c in data.NON_FEATURE:
        use = '제외(발행월 기준으로 사용)' if c == 'issue_d' else '제외(식별자·자유 텍스트·상수)'
    elif c in feats:
        use = '모델 입력(101개 중 하나)'
    else:
        use = '제외'
    key = RENAMED.get(c, c)
    by = pd.Series(m).groupby(yr).mean().reindex(YEARS)
    first = next((y for y in YEARS if by[y] < 0.5), None)
    mo, mr = ~m & res, m & res
    gap = bad[mr].mean() - bad[mo].mean() if mr.sum() >= 30 and mo.sum() >= 30 else None
    t = mtype.get(c)
    a13, a16 = avail[2013][c], avail[2016][c]
    tree = linear = None
    if use.startswith(('모델 입력', '입력(')) and t:
        tree, linear = '결측 그대로(트리가 스스로 처리)', LINEAR[t]
    rows.append({
        '현재 처리': use,
        '결측 유형': TYPE_KO.get(t),
        '결측 처리(트리)': tree,
        '결측 처리(선형)': linear,
        '값이 절반 넘게 있는 첫 해': first,
        '부도율 차이(결측-관측)': gap,
        '학습 가용률(2013 모델)': a13,
        '학습 가용률(2016 모델)': a16,
        '선형 모델 가용성 규칙(2013/2016)': ('/'.join('사용' if a >= 0.5 else '제외' for a in (a13, a16))
                                    if use.startswith(('모델 입력', '입력(')) and pd.api.types.is_numeric_dtype(raw[c]) else None),
        'IV(혼자서 부도를 가르는 힘)': iv['IV'].get(key) if key in iv.index else None,
        'IV 강도': iv['강도'].get(key) if key in iv.index else None,
        'LightGBM 중요도 순위(SHAP)': int(imp['순위'][key]) if key in imp.index else None,
        '방향': imp['방향'][key] if key in imp.index else None,
        '등급 안 정보 비율': wg['등급 안 정보 비율'].get(key) if key in wg.index else None,
        '순위상관 0.9 이상 짝': ', '.join(partner.get(key, [])) or None,
        '확인된 사실(1~26단계)': NOTE.get(c),
    })
NEW = pd.DataFrame(rows, index=raw.columns)

wb = load_workbook(SRC)
ws = wb['컬럼 사전']
head = [c.value for c in ws[1]]
start = len(head) + 1
hdr_font, hdr_fill = copy(ws.cell(1, 1).font), copy(ws.cell(1, 1).fill)
new_fill = PatternFill('solid', fgColor='FFE2EFDA')
pct = {'학습 가용률(2013 모델)', '학습 가용률(2016 모델)', '부도율 차이(결측-관측)'}
for j, col in enumerate(NEW.columns):
    cell = ws.cell(1, start + j, col)
    cell.font, cell.fill = hdr_font, hdr_fill
    cell.alignment = Alignment(wrap_text=True, vertical='center')
    ws.column_dimensions[get_column_letter(start + j)].width = 44 if col == '확인된 사실(1~26단계)' else (24 if '처리' in col or '유형' in col or '짝' in col else 14)
missing_cols = []
for i in range(2, ws.max_row + 1):
    name = ws.cell(i, 1).value
    if name not in NEW.index:
        missing_cols.append(name)
        continue
    for j, col in enumerate(NEW.columns):
        v = NEW.at[name, col]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        cell = ws.cell(i, start + j, v.item() if hasattr(v, 'item') else v)
        cell.fill = new_fill
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        cell.font = Font(name=ws.cell(i, 1).font.name or 'Arial', size=ws.cell(i, 1).font.size or 10)
        if col in pct:
            cell.number_format = '0.0%'
        elif col in ('값이 절반 넘게 있는 첫 해', 'LightGBM 중요도 순위(SHAP)'):
            cell.value = int(cell.value)
            cell.number_format = '0'
        elif col.startswith('IV(') or col == '등급 안 정보 비율':
            cell.number_format = '0.000'
ws.auto_filter.ref = f'A1:{get_column_letter(start + len(NEW.columns) - 1)}{ws.max_row}'

guide = wb['읽는 법']
r0 = guide.max_row + 2
lines = [
    ('갱신(2026-11-10)', '오른쪽 초록색 열은 1~26단계 실험으로 확인한 사실. 왼쪽 원래 열(팀원 작성)은 그대로 둠'),
    ('현재 처리', '지금 코드(src/lc/data.py)에서 실제로 하는 처리. 모델 입력은 101개'),
    ('결측 유형', '왜 비어 있나: 구조적, 사건 없음, 시기(새 변수/옛 대출), MNAR 의심, 소량'),
    ('결측 처리(트리/선형)', '22단계 결정. 트리(LightGBM)는 결측 그대로, 선형은 가용성 규칙 → 1~99% 자르기 → 중앙값'),
    ('학습 가용률', '그 해 모델의 학습 데이터에서 값이 있는 비율. 50% 미만이면 선형 모델에서 제외(가용성 규칙)'),
    ('부도율 차이(결측-관측)', '값이 비었을 때 부도율 - 값이 있을 때 부도율(종결 대출). 크면 결측 자체가 신호'),
    ('IV', '혼자서 부도를 가르는 힘. 0.02 미만 약함, 0.1~0.3 중간, 0.3 이상 강함'),
    ('중요도 순위', 'LightGBM 모델이 실제로 많이 쓴 순서(SHAP). 방향은 값이 클수록 위험이 오르는지'),
    ('등급 안 정보 비율', '같은 세부등급 안에서도 남는 신호의 비율. 1에 가깝거나 크면 등급이 덜 반영한 정보'),
    ('주의', "비고 열의 '미검증' 가설 중 결측 관련은 결측_검증시트.xlsx와 이 파일의 결측 열로 확인 가능"),
]
for k, (a, b) in enumerate(lines):
    guide.cell(r0 + k, 1, a).font = Font(bold=True)
    guide.cell(r0 + k, 2, b).alignment = Alignment(wrap_text=True)
OUT.parent.mkdir(exist_ok=True)
wb.save(OUT)
print('저장', OUT, '| 사전에 없는 데이터 컬럼', sorted(set(NEW.index) - {ws.cell(i, 1).value for i in range(2, ws.max_row + 1)}))
print('데이터에 없는 사전 행', missing_cols)
print(NEW['현재 처리'].value_counts().to_string())
print(NEW.loc[['annual_inc', 'il_util', 'emp_length', 'mths_since_last_delinq', 'int_rate', 'total_pymnt']].T.to_string())
