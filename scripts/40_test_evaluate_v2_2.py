"""test 평가 v2.2: v1, v2.1을 그대로 계산한 뒤(35번), docs/test_protocol_v2_2.md의 최종 제안 판정(H10~H12)을 덧붙인다.

사용:
  python scripts/40_test_evaluate_v2_2.py <test 파일>
  python scripts/40_test_evaluate_v2_2.py --dry-run
결과: results/test_v2_2/ (예행연습은 results/test_v2_2_dryrun/)
최종 제안은 39단계에서 train에 선택 규칙을 한 번 적용해 정했고, test를 본 뒤에는 바꾸지 않는다.
"""
import _path  # noqa: F401
from pathlib import Path

PICK = ('V2 부도확률 낮은 순', 0.8)  # 39단계 결과(태그 test-protocol-v2.2), test 후 변경 금지

SRC = (Path(__file__).parent / '35_test_evaluate_v2.py').read_text()
SRC = SRC.replace("'test_v2_1_dryrun' if DRY else 'test_v2_1'", "'test_v2_2_dryrun' if DRY else 'test_v2_2'")
exec(compile(SRC, '35_test_evaluate_v2.py', 'exec'))
assert PICK[1] is not None

# ===================== v2.2 =====================
QS22 = [i / 10 for i in range(1, 10)]
e = main
n = e['ym'].nunique()
st = np.random.default_rng(2026).integers(0, max(1, n - BLOCK + 1), size=(NB, int(np.ceil(n / BLOCK))))
BIX = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st[:, [j]] for j in range(st.shape[1])], axis=1)[:, :n], n - 1)
vsharpe = lambda x: x.mean(axis=-1) / x.std(axis=-1, ddof=1)
jit = np.random.default_rng(11).random(len(e)) * 1e-6
scores = {'V1 기대수익 순': e['er'].to_numpy() + jit, 'V2 부도확률 낮은 순': -e['pdx'].to_numpy() + jit}

rows = []
for q in QS22:
    for nm, sc in scores.items():
        port = monthly(e, by_month_top(e, sc, q))[1]
        rows.append({'전략': nm, '승인비율': q, 'Sharpe': sharpe(port), '평균 초과': port.mean()})
C = pd.DataFrame(rows)
C['순위'] = C['Sharpe'].rank(ascending=False).astype(int)

nm, qp = PICK
a_pick = by_month_top(e, scores[nm], qp)
ch = monthly(e, a_pick)[1]
rnd = np.array([monthly(e, by_month_top(e, np.random.default_rng(500 + k).random(len(e)), qp))[1] for k in range(10)])
diff = sharpe(ch) - np.mean([sharpe(r) for r in rnd])
dbs = vsharpe(ch[BIX]) - np.mean([vsharpe(r[BIX]) for r in rnd], axis=0)
lo, hi = float(np.quantile(dbs, 0.05)), float(np.quantile(dbs, 0.95))
all_s = sharpe(monthly(e, np.ones(len(e), bool))[1])
row = C[(C['전략'] == nm) & np.isclose(C['승인비율'], qp)].iloc[0]
tb0 = e['tb'].to_numpy(float)
hz = {'국채+1%p': tb0 + 0.01, '국채+2%p': tb0 + 0.02, 'BAA': e['baa'].to_numpy(float)}
hz_rep = {hn: (sharpe(monthly(e, a_pick, h)[1]), monthly(e, a_pick, h)[1].mean()) for hn, h in hz.items()}
v3 = sharpe(monthly(e, v2_rules(e, qp)['V3 등급 안 계좌 개설 적은 순'])[1])

h10 = lo > 0
h11 = bool(sharpe(ch) > all_s)
V22 = pd.DataFrame([
    ('H10 제안 유지: 선택안 - 무작위 Sharpe 90% 하한 > 0', h10, f'{diff:.2f} ({lo:.2f}~{hi:.2f})'),
    ('H11 전부 승인보다 Sharpe 높음(보고)', h11, f'{sharpe(ch):.2f} vs {all_s:.2f}'),
    ('H12 18개 중 순위(보고)', None, f"{int(row['순위'])}위, 1위와 차이 {C['Sharpe'].max() - row['Sharpe']:.2f}"),
    ('V3 같은 비율 Sharpe(도전자, 보고)', None, f'{v3:.2f}'),
], columns=['가설', '지지', '값'])
C.round(4).to_csv(OUT / 'V5_candidates_v2_2.csv', index=False)
V22.to_csv(OUT / 'verdict_v2_2.csv', index=False)
md2 = ['', '# v2.2 최종 제안 판정', f'최종 제안: {nm} 승인 {int(round(qp * 100))}% (39단계, test 전 고정)', '', md_table(V22), '',
       '## 허들별(보고)', md_table(pd.DataFrame([(k, v[0], v[1]) for k, v in hz_rep.items()], columns=['허들', 'Sharpe', '평균 초과'])), '',
       '## test의 18개 후보', md_table(C)]
with open(OUT / 'summary.md', 'a') as fh:
    fh.write('\n'.join(md2) + '\n')
print('\n'.join(md2[:6]))
