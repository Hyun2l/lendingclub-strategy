from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RAW_CSV = DATA / 'raw' / 'lending_club_2020_train.csv'
TRAIN_PARQUET = DATA / 'processed' / 'lc_train.parquet'
TREASURY = {36: DATA / 'external' / 'GS3.csv', 60: DATA / 'external' / 'GS5.csv'}
RESULTS = ROOT / 'results'
