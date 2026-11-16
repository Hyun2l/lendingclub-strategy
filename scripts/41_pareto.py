"""41단계: 수익·안정성 파레토 점검(팀원 검토: '둘 다 나쁜 조합은 빼고, 둘 다 좋아지는 조합을 찾고, 양립 안 되는 지점에서만 고르자').

축: 발행월 포트폴리오의 평균 초과수익(수익)과 표준편차(흔들림, 낮을수록 좋음). Sharpe = 평균 / 표준편차.
후보: V1 기대수익 순, V2 부도확률 낮은 순, V3 단순 규칙(등급 안 계좌 개설 적은 순, 사후 규칙), 무작위 x 승인 10~90%, 국채 허들.
지배: 다른 후보가 평균은 같거나 높고 표준편차는 같거나 낮으며, 둘 중 하나는 엄격히 나으면 '지배당함'.
판정 없음(보고). v2.2의 최종 제안(V2 80%)은 바꾸지 않는다.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '39_final_choice.py').read_text()
exec(SRC[:SRC.index("rows, series = [], {}")])

rows = []
for q in QS:
    rules = {'V1 기대수익 순': cut(er, q, 'ym'), 'V2 부도확률 낮은 순': cut(lowpd, q, 'ym'),
             'V3 단순 규칙': cut(-acc + jit, q, ['ym', 'grade'])}
    for nm, a in rules.items():
        p = port_series(a, TB)
        rows.append({'전략': nm, '승인비율': q, '평균 초과(%p)': 100 * p.mean(), '표준편차(%p)': 100 * p.std(ddof=1), 'Sharpe': sharpe(p)})
    rp = [port_series(cut(np.random.default_rng(500 + k).random(len(ev)), q, 'ym'), TB) for k in range(5)]
    rows.append({'전략': '무작위', '승인비율': q, '평균 초과(%p)': 100 * np.mean([p.mean() for p in rp]),
                 '표준편차(%p)': 100 * np.mean([p.std(ddof=1) for p in rp]), 'Sharpe': np.mean([sharpe(p) for p in rp])})
    print('승인', q, f'{(time.time() - t0) / 60:.1f}분', flush=True)
T = pd.DataFrame(rows)


def dominated(df):
    m, s = df['평균 초과(%p)'].to_numpy(), df['표준편차(%p)'].to_numpy()
    return np.array([bool(((m >= m[i]) & (s <= s[i]) & ((m > m[i]) | (s < s[i]))).any()) for i in range(len(df))])


T['지배당함(전체)'] = dominated(T)
noV3 = T['전략'] != 'V3 단순 규칙'
T['지배당함(V3 제외)'] = False
T.loc[noV3, '지배당함(V3 제외)'] = dominated(T[noV3].reset_index(drop=True))
T.round(4).to_csv(RESULTS / '41_pareto.csv', index=False)
pd.set_option('display.width', 200)
for c in ['평균 초과(%p)', '표준편차(%p)', 'Sharpe']:
    print(f'\n== {c}')
    print(T.pivot(index='승인비율', columns='전략', values=c).round(2).to_string())
for col in ['지배당함(V3 제외)', '지배당함(전체)']:
    sub = T if col == '지배당함(전체)' else T[noV3]
    f = sub[~sub[col]].sort_values('표준편차(%p)')
    print(f'\n== 파레토 경계({col.replace("지배당함", "").strip("()")}):')
    print(f[['전략', '승인비율', '평균 초과(%p)', '표준편차(%p)', 'Sharpe']].round(2).to_string(index=False))
pk = T[(T['전략'] == 'V2 부도확률 낮은 순') & np.isclose(T['승인비율'], 0.8)].iloc[0]
print(f"\nv2.2 제안(V2 80%): 지배당함(V3 제외)={pk['지배당함(V3 제외)']}, 지배당함(전체)={pk['지배당함(전체)']}")
