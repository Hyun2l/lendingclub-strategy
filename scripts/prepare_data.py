"""원본 CSV를 한 번 읽어 parquet 캐시로 저장한다."""
import _path  # noqa: F401
import pandas as pd
from lc.config import RAW_CSV, TRAIN_PARQUET

if TRAIN_PARQUET.exists():
    print('이미 있음:', TRAIN_PARQUET)
else:
    TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df.to_parquet(TRAIN_PARQUET)
    print('저장:', TRAIN_PARQUET, df.shape)
