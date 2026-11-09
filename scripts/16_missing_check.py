"""16단계: 팀원이 세운 결측 원인 가설을 데이터로 확인하는 검증 시트(엑셀)를 만든다.

사용: python scripts/16_missing_check.py [컬럼사전.xlsx 경로]
출력: docs/결측_검증시트.xlsx

모든 숫자는 train 데이터(빈 행 1개 제외 1,755,294건)에서 이 스크립트로 계산한 값이다.
부도율은 종결 대출(정상 상환·부도, 신용정책 미충족 제외 1,116,156건) 기준이다.
"""
import sys
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from lc import data
from lc.config import ROOT

DICT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Downloads' / 'lending_club_컬럼사전.xlsx'
OUT = ROOT / 'docs' / '결측_검증시트.xlsx'
YEARS = list(range(2007, 2021))
MIN_N = 30  # 이보다 적으면 부도율을 비워 둔다

# ---------- 계산 ----------
raw = data.load()
d = data.parse(raw)
cols = [c for c in raw.columns if raw[c].isna().any()]
miss = raw[cols].isna()
res = (d['resolved'] & ~d['policy_fail']).to_numpy()
bad = d['bad'].to_numpy()
yr = d['issue_year'].to_numpy()
joint = (d['application_type'].astype(str) == 'Joint App').to_numpy()

dic = {}
if DICT.exists():
    ws = load_workbook(DICT, read_only=True)['컬럼 사전']
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    for r in rows[1:]:
        rec = dict(zip(head, r))
        dic[rec['컬럼명']] = rec

main, yearly = [], []
for c in cols:
    m = miss[c].to_numpy()
    mr, mo = m & res, ~m & res
    row = {'컬럼명': c, '그룹': (dic.get(c) or {}).get('그룹', ''), '구분': (dic.get(c) or {}).get('구분', ''),
           '한글 설명': (dic.get(c) or {}).get('한글 설명', ''), '결측 건수': int(m.sum()), '결측률(전체)': m.mean(),
           '결측률(종결 대출)': m[res].mean(),
           '부도율: 값이 있을 때': bad[mo].mean() if mo.sum() >= MIN_N else None,
           '부도율: 결측일 때': bad[mr].mean() if mr.sum() >= MIN_N else None,
           '결측률: 개인 신청': m[~joint].mean(), '결측률: 공동 신청': m[joint].mean()}
    if row['부도율: 값이 있을 때'] is not None and row['부도율: 결측일 때'] is not None:
        row['부도율 차이(결측-관측)'] = row['부도율: 결측일 때'] - row['부도율: 값이 있을 때']
    else:
        row['부도율 차이(결측-관측)'] = None
    by = pd.Series(m).groupby(yr).mean().reindex(YEARS)
    row['결측률 가장 높은 해'] = int(by.idxmax())
    row['결측률 가장 낮은 해'] = int(by.idxmin())
    row['연도 간 차이(최고-최저)'] = float(by.max() - by.min())
    main.append(row)
    yearly.append({'컬럼명': c, **{str(y): by[y] for y in YEARS}})
M = pd.DataFrame(main).sort_values('결측률(전체)', ascending=False).reset_index(drop=True)
Y = pd.DataFrame(yearly).set_index('컬럼명').loc[M['컬럼명']].reset_index()

# 같이 비는 컬럼 묶음: 결측 여부의 상관 0.95 이상을 연결
mid = [c for c in cols if 0.001 < miss[c].mean() < 0.999]
C = miss[mid].astype('int8').corr().to_numpy()
parent = list(range(len(mid)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(mid)):
    for j in range(i + 1, len(mid)):
        if C[i, j] >= 0.95:
            parent[find(i)] = find(j)
groups = {}
for i, c in enumerate(mid):
    groups.setdefault(find(i), []).append(c)
G = [g for g in groups.values() if len(g) >= 2]
G = sorted(G, key=lambda g: -miss[g].mean().mean())
gid = {c: f'묶음 {k + 1}' for k, g in enumerate(G) for c in g}
M['같이 비는 묶음'] = M['컬럼명'].map(gid).fillna('')
GT = pd.DataFrame([{'묶음': f'묶음 {k + 1}', '컬럼 수': len(g), '평균 결측률': float(miss[g].mean().mean()),
                    '그룹(사전 기준)': ', '.join(sorted({str((dic.get(c) or {}).get('그룹', '')) for c in g})),
                    '컬럼': ', '.join(g)} for k, g in enumerate(G)])

# ---------- 엑셀 ----------
F = 'Arial'
HEAD_FILL = PatternFill('solid', fgColor='1F3A5F')
FILL_IN = PatternFill('solid', fgColor='FFF2CC')
thin = Side(style='thin', color='D0D7DE')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook()


def style_header(ws, ncol, row=1):
    for j in range(1, ncol + 1):
        c = ws.cell(row=row, column=j)
        c.font = Font(name=F, bold=True, color='FFFFFF', size=10)
        c.fill = HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
        c.border = BORDER


# 1) 읽는 법
g = wb.active
g.title = '읽는 법'
guide = [
    ('결측 검증 시트', ''),
    ('목적', '컬럼 사전을 보고 세운 결측 원인 가설이 데이터와 맞는지 확인하는 시트입니다. 먼저 가설을 쓰고, 그다음 숫자를 보세요.'),
    ('순서', '1) "결측 검증" 시트에서 맡은 컬럼을 찾는다  2) 노란 칸에 가설과 이유를 먼저 쓴다  3) 회색 숫자 칸으로 맞는지 확인한다  4) 결과와 처리 제안을 쓴다'),
    ('데이터 출처', 'train 데이터 1,755,294건(빈 행 1개 제외). 부도율은 종결 대출 1,116,156건 기준. 모든 숫자는 scripts/16_missing_check.py로 계산한 값입니다.'),
    ('', ''),
    ('숫자 칸 읽는 법', ''),
    ('결측률(전체)', '이 컬럼이 비어 있는 비율'),
    ('부도율: 값이 있을 때 / 결측일 때', '종결 대출 중 이 컬럼이 비어 있는 대출과 아닌 대출의 부도율. 결측 건수가 30건 미만이면 비워 둠'),
    ('부도율 차이(결측-관측)', '양수면 비어 있는 대출이 더 위험. 차이가 크면 "결측 자체가 정보"라는 뜻'),
    ('결측률: 개인 / 공동 신청', '공동 신청에서만 값이 있으면 개인 쪽 결측률이 100%에 가까움 → 구조적 결측 단서'),
    ('결측률 가장 높은·낮은 해, 연도 간 차이', '특정 해부터 갑자기 값이 생기거나 사라지면 시기(제도 변경) 때문일 가능성 → "연도별 결측률" 시트에서 확인. 같은 값인 해가 여러 개면 가장 이른 해를 표시'),
    ('같이 비는 묶음', '항상 같은 대출에서 함께 비는 컬럼들. 같은 원인일 가능성이 큼 → "같이 비는 묶음" 시트에서 확인'),
    ('', ''),
    ('결측 유형(노란 칸에 고르기)', ''),
    ('구조적', '해당 사항이 없어서 빈 것 (예: 공동 신청이 아니면 sec_app_* 가 빔)'),
    ('MCAR', '완전히 우연히 빠짐. 부도율 차이도, 연도 차이도 거의 없음'),
    ('MAR', '다른 관측된 변수로 설명되는 결측 (예: 나중에 생긴 변수라 옛날 대출만 빔 → 발행 연도로 설명됨)'),
    ('MNAR', '빠진 값 자체 때문에 비어 있음 (예: 무직이라 근속 연수를 안 씀). 데이터만으로는 증명할 수 없으니 "의심" 수준으로 씀'),
    ('누수', '대출 이후에 생기는 정보라 결측 여부 자체가 미래 정보 (예: hardship_*)'),
    ('', ''),
    ('채우는 예시', '컬럼 mths_since_last_delinq → 유형: 구조적 / 이유: 연체 이력이 없는 사람은 "마지막 연체 이후 개월"이 없음 / 데이터와 맞나: 맞음(결측일 때 부도율이 낮음) / 처리: 결측 표시 컬럼 추가 후 큰 값으로 채우기'),
]
for i, (a, b) in enumerate(guide, start=1):
    g.cell(row=i, column=1, value=a).font = Font(name=F, bold=(b == '' and a != '') or a in ('목적', '순서', '데이터 출처'), size=10)
    c = g.cell(row=i, column=2, value=b)
    c.font = Font(name=F, size=10)
    c.alignment = Alignment(wrap_text=True, vertical='top')
g.column_dimensions['A'].width = 34
g.column_dimensions['B'].width = 110
g.cell(row=1, column=1).font = Font(name=F, bold=True, size=14)

# 2) 결측 검증
w = wb.create_sheet('결측 검증')
info_cols = ['컬럼명', '그룹', '구분', '한글 설명', '결측 건수', '결측률(전체)', '결측률(종결 대출)', '부도율: 값이 있을 때', '부도율: 결측일 때',
             '부도율 차이(결측-관측)', '결측률: 개인 신청', '결측률: 공동 신청', '결측률 가장 높은 해', '결측률 가장 낮은 해', '연도 간 차이(최고-최저)', '같이 비는 묶음']
fill_cols = ['내 가설: 결측 유형', '그렇게 본 이유', '데이터와 맞나?', '처리 제안', '질문·메모']
headers = info_cols + fill_cols
for j, h in enumerate(headers, start=1):
    w.cell(row=1, column=j, value=h)
style_header(w, len(headers))
for j in range(len(info_cols) + 1, len(headers) + 1):
    w.cell(row=1, column=j).fill = PatternFill('solid', fgColor='B7791F')
pct = {'결측률(전체)', '결측률(종결 대출)', '부도율: 값이 있을 때', '부도율: 결측일 때', '결측률: 개인 신청', '결측률: 공동 신청', '연도 간 차이(최고-최저)'}
for i, r in M.iterrows():
    for j, h in enumerate(info_cols, start=1):
        v = r[h]
        if isinstance(v, float) and np.isnan(v):
            v = None
        c = w.cell(row=i + 2, column=j, value=v)
        c.font = Font(name=F, size=10)
        c.border = BORDER
        c.alignment = Alignment(vertical='top', wrap_text=(h == '한글 설명'))
        if h in pct:
            c.number_format = '0.0%'
        elif h == '부도율 차이(결측-관측)':
            c.number_format = '+0.0%;-0.0%;0.0%'
        elif h == '결측 건수':
            c.number_format = '#,##0'
    for j in range(len(info_cols) + 1, len(headers) + 1):
        c = w.cell(row=i + 2, column=j)
        c.fill = FILL_IN
        c.border = BORDER
        c.font = Font(name=F, size=10)
        c.alignment = Alignment(vertical='top', wrap_text=True)
n = len(M) + 1
widths = {'컬럼명': 30, '그룹': 22, '구분': 18, '한글 설명': 44, '결측 건수': 11, '같이 비는 묶음': 11, '내 가설: 결측 유형': 14,
          '그렇게 본 이유': 36, '데이터와 맞나?': 13, '처리 제안': 28, '질문·메모': 28}
for j, h in enumerate(headers, start=1):
    w.column_dimensions[get_column_letter(j)].width = widths.get(h, 12)
w.row_dimensions[1].height = 42
w.freeze_panes = 'B2'
w.auto_filter.ref = f'A1:{get_column_letter(len(headers))}{n}'
col_of = {h: get_column_letter(j) for j, h in enumerate(headers, start=1)}
w.conditional_formatting.add(f"{col_of['부도율 차이(결측-관측)']}2:{col_of['부도율 차이(결측-관측)']}{n}",
                             ColorScaleRule(start_type='num', start_value=-0.1, start_color='63BE7B', mid_type='num', mid_value=0,
                                            mid_color='FFFFFF', end_type='num', end_value=0.1, end_color='F8696B'))
w.conditional_formatting.add(f"{col_of['연도 간 차이(최고-최저)']}2:{col_of['연도 간 차이(최고-최저)']}{n}",
                             ColorScaleRule(start_type='num', start_value=0, start_color='FFFFFF', end_type='num', end_value=1, end_color='5B8BD9'))
dv1 = DataValidation(type='list', formula1='"구조적,MCAR,MAR,MNAR,누수,모르겠음"', allow_blank=True)
dv2 = DataValidation(type='list', formula1='"맞음,아님,일부만 맞음,모르겠음"', allow_blank=True)
w.add_data_validation(dv1)
w.add_data_validation(dv2)
dv1.add(f"{col_of['내 가설: 결측 유형']}2:{col_of['내 가설: 결측 유형']}{n}")
dv2.add(f"{col_of['데이터와 맞나?']}2:{col_of['데이터와 맞나?']}{n}")

# 3) 연도별 결측률
yv = wb.create_sheet('연도별 결측률')
yh = ['컬럼명'] + [str(y) for y in YEARS]
for j, h in enumerate(yh, start=1):
    yv.cell(row=1, column=j, value=h)
style_header(yv, len(yh))
for i, r in Y.iterrows():
    for j, h in enumerate(yh, start=1):
        v = r[h]
        if isinstance(v, float) and np.isnan(v):
            v = None
        c = yv.cell(row=i + 2, column=j, value=v)
        c.font = Font(name=F, size=10)
        c.border = BORDER
        if j > 1:
            c.number_format = '0%'
yv.column_dimensions['A'].width = 34
for j in range(2, len(yh) + 1):
    yv.column_dimensions[get_column_letter(j)].width = 7
yv.freeze_panes = 'B2'
yv.conditional_formatting.add(f'B2:{get_column_letter(len(yh))}{len(Y) + 1}',
                              ColorScaleRule(start_type='num', start_value=0, start_color='FFFFFF', end_type='num', end_value=1, end_color='5B8BD9'))
yv.cell(row=len(Y) + 3, column=1, value='발행연도별로 이 컬럼이 비어 있는 비율. 진한 파랑일수록 많이 빔. 2007~2008년은 대출 수가 적어 흔들릴 수 있음.').font = Font(name=F, size=9, italic=True)

# 4) 같이 비는 묶음
gs = wb.create_sheet('같이 비는 묶음')
gh = ['묶음', '컬럼 수', '평균 결측률', '그룹(사전 기준)', '컬럼']
for j, h in enumerate(gh, start=1):
    gs.cell(row=1, column=j, value=h)
style_header(gs, len(gh))
for i, r in GT.iterrows():
    for j, h in enumerate(gh, start=1):
        c = gs.cell(row=i + 2, column=j, value=r[h])
        c.font = Font(name=F, size=10)
        c.border = BORDER
        c.alignment = Alignment(vertical='top', wrap_text=(h == '컬럼'))
        if h == '평균 결측률':
            c.number_format = '0.0%'
for j, wd in enumerate([9, 9, 12, 30, 110], start=1):
    gs.column_dimensions[get_column_letter(j)].width = wd
gs.cell(row=len(GT) + 3, column=1, value='결측 여부의 상관이 0.95 이상인 컬럼끼리 묶었음. 같은 묶음은 같은 대출에서 거의 항상 함께 비어 있어, 같은 원인일 가능성이 큼.').font = Font(name=F, size=9, italic=True)

OUT.parent.mkdir(exist_ok=True)
wb.save(OUT)
print('저장:', OUT, '| 결측 있는 컬럼', len(M), '| 같이 비는 묶음', len(G), '| 사전 매칭', int((M['그룹'] != '').sum()))
print(M[['컬럼명', '결측률(전체)', '부도율 차이(결측-관측)', '결측률 가장 높은 해', '결측률 가장 낮은 해', '같이 비는 묶음']].head(12).round(3).to_string(index=False))
print(GT[['묶음', '컬럼 수', '평균 결측률', '그룹(사전 기준)']].round(3).to_string(index=False))
