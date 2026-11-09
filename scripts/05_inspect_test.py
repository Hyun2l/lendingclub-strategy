"""5단계: test 데이터를 받으면 가장 먼저 실행한다.

사용: python scripts/05_inspect_test.py <test 파일 경로(.csv 또는 .parquet)>
확인: 컬럼 차이, 사후 컬럼 유무, 결과 상태 분포, 발행연도 분포(train 대비), train과 id 중복.
"""
import sys
from pathlib import Path

import _path  # noqa: F401
import pandas as pd
import pyarrow.parquet as pq
from lc import data
from lc.config import TRAIN_PARQUET, RESULTS

path = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_PARQUET
read = (lambda c=None: pd.read_parquet(path, columns=c)) if path.suffix == '.parquet' else \
       (lambda c=None: pd.read_csv(path, usecols=c, low_memory=False))
test_cols = pq.read_schema(path).names if path.suffix == '.parquet' else list(pd.read_csv(path, nrows=0).columns)
train_cols = pq.read_schema(TRAIN_PARQUET).names
lines = [f'파일: {path}', f'컬럼 수: test {len(test_cols)} / train {len(train_cols)}',
         f'test에만 있는 컬럼: {sorted(set(test_cols) - set(train_cols))}',
         f'train에만 있는 컬럼: {sorted(set(train_cols) - set(test_cols))}']
post = [c for c in ['loan_status', 'total_pymnt', 'last_pymnt_d', 'recoveries', 'out_prncp'] if c in test_cols]
lines.append(f'사후 컬럼(수익 계산에 필요): {post or "없음 -> 우리가 test 수익을 계산할 수 없다"}')

need = [c for c in ['id', 'issue_d', 'loan_status', 'term'] if c in test_cols]
t = read(need)
lines.append(f'행 수: {len(t):,}')
if 'loan_status' in t:
    s = data.classify(t['loan_status'].dropna()).value_counts()
    lines.append('결과 상태: ' + ', '.join(f'{k} {v:,}' for k, v in s.items()))
    lines.append(f"진행 중(current+late) 비율: {(s.get('current', 0) + s.get('late', 0)) / s.sum():.3f}")
tr = pd.read_parquet(TRAIN_PARQUET, columns=['id', 'issue_d'])
yr = pd.DataFrame({
    'test': pd.to_datetime(t['issue_d'], format='%b-%Y', errors='coerce').dt.year.value_counts(),
    'train': pd.to_datetime(tr['issue_d'], format='%b-%Y', errors='coerce').dt.year.value_counts()}).fillna(0).sort_index()
yr['test/train'] = yr['test'] / yr['train']
yr['test 비중'] = yr['test'] / yr['test'].sum()
lines.append('발행연도 분포 (test/train 비율이 연도마다 비슷하면 같은 기간에서 무작위로 나눈 것):')
lines.append(yr.round(3).to_string())
if 'id' in t:
    ov = t['id'].astype(str).isin(set(tr['id'].astype(str))).mean()
    lines.append(f'train과 id 중복 비율: {ov:.3f} (0이어야 정상)')
out = '\n'.join(lines)
print(out)
RESULTS.mkdir(exist_ok=True)
(RESULTS / f'05_inspect_{path.stem}.txt').write_text(out + '\n')
