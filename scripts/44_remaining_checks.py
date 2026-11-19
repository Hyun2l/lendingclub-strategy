"""44단계: 팀 검토 문서가 요청한 남은 점검(보고용, test 규칙 불변).

1. 금리 높은 순(모델 없음) 승인 10~90%: 기대수익 순의 '공정한 짝'
2. s > 0 전부 승인(발행월별 승인 수 = s > 0 건수), 같은 건수 무작위와 비교
3. 승인 80%에서 V1 - V2 선별 가치 차이의 90% 구간
방식은 42단계와 같다(발행월 컷, 국채 허들, 무작위 10회, 6개월 블록 부트스트랩).
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '42_final_proposal_check.py').read_text()
exec(SRC[:SRC.index("rows = []")])

hirate = ev['int_rate'].astype(str).str.strip().str.rstrip('%').astype(float).to_numpy() + jit


def topk(score, k):
    a = np.zeros(len(ev), bool)
    for m in months:
        ix = grp[m]
        if k[m] > 0:
            a[ix[np.argsort(-score[ix])[:k[m]]]] = True
    return a


def summary(nm, q, a, rnd):
    rp = np.array([x[0] for x in rnd]); rr = np.array([x[1] for x in rnd]).mean(axis=0)
    rs, rs_bs = np.mean([sharpe(x) for x in rp]), np.mean([sharpe(x[BIX]) for x in rp], axis=0)
    p, r = series(a)
    sv = r - rr
    svb = sv[BIX].mean(axis=1); d_bs = sharpe(p[BIX]) - rs_bs
    dr = np.mean([ev['def'].to_numpy()[a].mean()]) if 'def' in ev else np.nan
    return {'전략': nm, '승인비율': q, '선별 가치(%p)': 100 * sv.mean(), '선별 하한': 100 * q90(svb)[0],
            '선별 상한': 100 * q90(svb)[1], 'Sharpe': sharpe(p), '무작위 Sharpe': rs,
            '차이 하한': q90(d_bs)[0], '차이 상한': q90(d_bs)[1], '평균 초과(%p)': 100 * p.mean()}, r


rows = []
for q in QS:
    rnd = [series(cut(np.random.default_rng(500 + k).random(len(ev)), q)) for k in range(10)]
    row, _ = summary('금리 높은 순', q, cut(hirate, q), rnd)
    rows.append(row)
    if abs(q - 0.8) < 1e-9:
        _, r1 = summary('V1', q, cut(er, q), rnd)
        _, r2 = summary('V2', q, cut(lowpd, q), rnd)
        d = r1 - r2; db = d[BIX].mean(axis=1)
        print(f'80% V1-V2 선별 가치 차이 {100*d.mean():.3f}%p, 90% 구간 {100*q90(db)[0]:.3f}~{100*q90(db)[1]:.3f}')
    print('승인', q, flush=True)

spos = ev['s_er'].to_numpy() > 0
k = {m: int(spos[grp[m]].sum()) for m in months}
share = np.mean([k[m] / len(grp[m]) for m in months])
rnd = [series(topk(np.random.default_rng(500 + j).random(len(ev)), k)) for j in range(10)]
row, _ = summary('s > 0 전부 승인', round(share, 3), spos, rnd)
rows.append(row)
print(f's>0 승인 건수 비율: 평균 {share:.3f}, 최소 {min(k[m]/len(grp[m]) for m in months):.3f}, 최대 {max(k[m]/len(grp[m]) for m in months):.3f}')
C = pd.DataFrame(rows)
C.round(4).to_csv(RESULTS / '44_remaining_checks.csv', index=False)
pd.set_option('display.width', 220)
print(C.round(2).to_string(index=False))
