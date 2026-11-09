# LendingClub 대출 승인 전략

서울대 ABS 14기 팀 프로젝트. 투자자 입장에서 어떤 LendingClub 대출을 승인하고 어떤 대출 대신 국채에 투자할지 정하고, 그 선택의 근거를 검증한다.

## 데이터 준비
데이터는 용량 때문에 저장소에 포함하지 않는다.
1. 배포받은 `lending_club_2020_train.csv`를 `data/raw/`에 넣는다.
2. 미 국채 3년·5년 월별 금리는 `data/external/GS3.csv`, `GS5.csv`에 포함되어 있다(출처: FRED).
3. `python scripts/prepare_data.py`를 실행하면 `data/processed/lc_train.parquet`가 만들어진다.

환경: Python 3.11, `pip install -r requirements.txt`

## 폴더
| 위치 | 내용 |
|---|---|
| `src/lc/` | 팀 공통 코드(데이터 정리, 수익률, 기준선, 시점 학습) |
| `scripts/` | 단계별 실행 스크립트 |
| `tests/` | 손계산 예시로 확인하는 테스트 (`pytest`) |
| `results/` | 공통 파이프라인 결과 |
| `notebooks/` | 탐색 노트북과 초기 실험 |
| `notebooks/evidence/` | 방향을 정하기 위한 증거 실험. 요약은 `RESULTS.md` |

## 진행 단계
| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 저장소 구조, 초기 실험 보관 | 완료 |
| 2 | 데이터 정리: 누수 제거, 표본 정의 | 완료 |
| 3 | 수익률 함수, 국채 연결 | 완료 |
| 4 | 기준선 전략 표 | 완료 |
| 5 | test 점검 코드 | 완료 |
| 6 | 그 시점 정보만 쓰는 학습 틀 | 완료 |
| 7 | 기본 모델 비교 | 완료 (baseline-models 브랜치) |
| 8 | 비교 기준 바꾸기 | 완료 (rf-alternatives 브랜치) |
| 9 | 부도확률 보정 | 완료 (calibration 브랜치) |
| 10 | 은행 관점 RAROC 시험 | 완료 (raroc 브랜치) |
| 11 | 새 비교 방법 검증 | 완료 (novel-metrics 브랜치) |
| 12~13 | EDA와 피처 엔지니어링 | 완료 (feature-engineering 브랜치) |
| 14 | 비교 방법 재검증 | 완료 (validate-methods 브랜치) |
| 15 | test 평가 규칙 고정 | 완료 (test-protocol 브랜치, 태그 test-protocol-v1) |
| 16~17 | 결측 검증 시트, 피처 중요도·정리 문서 | 완료 |
| 18~19 | 후보 A 심화 검증(탐색) | 완료 (within-grade 브랜치) |

## 결정 기록
각 단계에서 무엇을 왜 골랐는지 `docs/decisions.md`에 남긴다.

## 실행 순서
```
python scripts/prepare_data.py
python -m pytest tests
python scripts/02_check_data.py
python scripts/03_returns.py
python scripts/04_baselines.py
python scripts/05_inspect_test.py <test 파일>
python scripts/06_asof_demo.py
```
