"""test 평가 v2.4: v1~v2.2를 그대로 계산한 뒤(40번), docs/test_protocol_v2_4.md의 주 판정(H14)과 보고(H15, H16, 민감도)를 덧붙인다.

사용:
  python scripts/47_test_evaluate_v2_4.py <test 파일>
  python scripts/47_test_evaluate_v2_4.py --dry-run
결과: results/test_v2_4/ (예행연습은 results/test_v2_4_dryrun/)
최종 제안은 46단계에서 train에 선택 규칙을 한 번 적용해 정했고, test를 본 뒤에는 바꾸지 않는다.
"""
import _path  # noqa: F401
from pathlib import Path

PICK_V24 = ('V2 부도확률 낮은 순', 0.4)  # 46단계 결과(docs/test_protocol_v2_4.md 7절), test 후 변경 금지

SRC = (Path(__file__).parent / '40_test_evaluate_v2_2.py').read_text()
SRC = SRC.replace("'test_v2_2_dryrun' if DRY else 'test_v2_2'", "'test_v2_4_dryrun' if DRY else 'test_v2_4'")
exec(compile(SRC, '40_test_evaluate_v2_2.py', 'exec'))
assert PICK_V24 is not None, '46단계 선택 결과를 PICK_V24에 먼저 적는다'

# ===================== v2.4 =====================
from lc import evaluate  # noqa: E402

e = main
ne = len(e)
lgd_e = e['issue_year'].map(LGD).to_numpy(float)
tb_e = e['tb'].to_numpy(float)
P_e = e['funded_amnt'].to_numpy(float)
X24 = np.minimum(returns.loan_irr(e, fee=FEE).to_numpy(), (1 + e['r'].to_numpy() / 12) ** 12 - 1) - tb_e  # 약정 IRR 상한(v2.4 6절)
X24_RE = loan_level(e)[1]
jit24 = np.random.default_rng(11).random(ne) * 1e-6
S24 = {'V1 기대수익 순': e['er'].to_numpy() - tb_e + jit24,
       'V2 부도확률 낮은 순': -e['pdx'].to_numpy() + jit24,
       'V4 위험조정 순': evaluate.risk_adjusted_score(e['pdx'], e['r'], lgd_e, tb_e) + jit24}
Q24 = [i / 10 for i in range(1, 10)]

mon24 = np.array(sorted(e['ym'].unique()))
MI24 = np.zeros(ne, int)
for k_, (m_, ix_) in enumerate(sorted(e.groupby('ym').indices.items())):
    MI24[ix_] = k_
NM24 = len(mon24)
st24 = np.random.default_rng(2026).integers(0, max(1, NM24 - BLOCK + 1), size=(NB, int(np.ceil(NM24 / BLOCK))))
BIX24 = np.minimum(np.concatenate([np.arange(BLOCK)[None, :] + st24[:, [j]] for j in range(st24.shape[1])], axis=1)[:, :NM24], NM24 - 1)


def boot24(a):
    x = np.where(a, X24, 0.0)
    S1, S2, N = (np.bincount(MI24, x, NM24), np.bincount(MI24, x * x, NM24), np.bincount(MI24, minlength=NM24).astype(float))
    s1, s2, nn = S1[BIX24].sum(1), S2[BIX24].sum(1), N[BIX24].sum(1)
    return (s1 / nn) / np.sqrt((s2 - s1 * s1 / nn) / (nn - 1))


def wsh(x, w):
    mu = (w * x).sum() / w.sum()
    return mu / np.sqrt((w * (x - mu) ** 2).sum() / w.sum())


rows24, A24 = [], {}
for nm_, sc_ in S24.items():
    for q in Q24:
        A24[(nm_, q)] = by_month_top(e, sc_, q)
A24[('전부 승인', 1.0)] = np.ones(ne, bool)
for (nm_, q), a_ in A24.items():
    rows24.append({'전략': nm_, '승인비율': q, '건별 Sharpe(주)': evaluate.loan_sharpe(X24, a_),
                   '건별 Sharpe(국채 재투자)': evaluate.loan_sharpe(X24_RE, a_), '건별 Sharpe(금액 가중)': wsh(np.where(a_, X24, 0.0), P_e),
                   '포트폴리오 Sharpe': sharpe(monthly(e, a_)[1])})
C24 = pd.DataFrame(rows24)
C24['순위'] = C24['건별 Sharpe(주)'].rank(ascending=False).astype(int)

nm24, q24 = PICK_V24
a24 = A24[(nm24, q24)]
s_pick = evaluate.loan_sharpe(X24, a24)
bp24 = boot24(a24)
if nm24 == '전부 승인':
    rnd24 = [a24]
else:
    rnd24 = [by_month_top(e, np.random.default_rng(500 + k).random(ne), q24) for k in range(10)]
s_rnd = np.mean([evaluate.loan_sharpe(X24, a) for a in rnd24])
br24 = np.mean([boot24(a) for a in rnd24], axis=0)
lo14, hi14 = np.quantile(bp24 - br24, [0.025, 0.975])
s_all = evaluate.loan_sharpe(X24, A24[('전부 승인', 1.0)])
lo15, hi15 = np.quantile(bp24 - boot24(A24[('전부 승인', 1.0)]), [0.025, 0.975])
r24 = C24[(C24['전략'] == nm24) & np.isclose(C24['승인비율'], q24)].iloc[0]

V24 = pd.DataFrame([
    ('H14 제안 유지(주 판정): 선택안 - 같은 비율 무작위 건별 Sharpe 95% 하한 > 0', bool(lo14 > 0), f'{s_pick - s_rnd:.4f} ({lo14:.4f}~{hi14:.4f})'),
    ('H15 전부 승인 대비(보고)', None, f'{s_pick:.4f} vs {s_all:.4f}, 차이 95% ({lo15:.4f}~{hi15:.4f})'),
    ('H16 28개 중 순위(보고)', None, f"{int(r24['순위'])}위, 1위와 차이 {C24['건별 Sharpe(주)'].max() - s_pick:.4f}"),
    ('민감도: 국채 재투자 / 금액 가중 / 포트폴리오(보고)', None,
     f"{r24['건별 Sharpe(국채 재투자)']:.4f} / {r24['건별 Sharpe(금액 가중)']:.4f} / {r24['포트폴리오 Sharpe']:.3f}"),
], columns=['가설', '지지', '값'])
C24.round(5).to_csv(OUT / 'V7_candidates_v2_4.csv', index=False)
V24.to_csv(OUT / 'verdict_v2_4.csv', index=False)
md4 = ['', '# v2.4 주 판정(교수님식 건별 Sharpe)', f'최종 제안: {nm24} 승인 {int(round(q24 * 100))}% (46단계, test 전 고정)', '',
       md_table(V24), '', '## test의 28개 후보', md_table(C24)]
with open(OUT / 'summary.md', 'a') as fh:
    fh.write('\n'.join(md4) + '\n')
print('\n'.join(md4[:6]))
