"""33단계(검증): Sortino, NPV, 등급 중립 롱숏 스프레드.

사전 비판(실행 전에 적음):
- Sortino: 발행월 포트폴리오는 48개월 모두 국채보다 높아(30단계) 아래쪽 편차가 0이 되므로 계산 불가. 그래서 승인 대출 건별로만 계산한다
  (대출 수익률 = 국채 재투자 만기 수익률, 기준선 0 = 국채와 같음). 대출 수익은 위가 막히고 아래가 열린 비대칭이라 Sharpe와 순위가 다를 수는 있으나,
  아래쪽 위험이 곧 부도라 부도확률 낮은 순이 다시 1위일 가능성이 크다(새 정보가 적을 위험).
- NPV: 국채로 할인한 순현재가치. 국채 재투자 만기 수익률과 같은 정보를 돈 단위로 바꾼 것이라 순위는 그것과 같다. 가치는 '규모'를 보여 주는 데 있다.
- 롱숏: LC 대출은 공매도 불가. 실제 전략이 아니라 '신호의 크기'를 재는 분석 도구다. 등급마다 같은 비율을 롱·숏하면 등급 배분과 시장 효과가 지워져
  '등급 안 선별'만 남는다(후보 A의 금융식 검증). 실제로 가능한 형태는 롱온리 비중 조절(상위 30% - 그 등급 전체).

판정(사전 고정):
- 롱숏 신호 인정: 등급 중립 스프레드의 이동 블록 부트스트랩(6개월) 90% 구간 하한 > 0 이고 양수인 발행월이 75% 이상. 무작위 신호(위약)는 구간이 0을 포함해야 한다.
- Sortino의 추가 정보: 승인비율 9개 중 3개 이상에서 Sortino 순위(4개 전략)가 건별 Sharpe 순위와 다르면 '추가 정보 있음', 아니면 'Sharpe와 같은 정보'.
- 등급 간 공정성: 등급별 발행월 초과수익의 Sharpe가 A~G에서 단조롭게 움직이지 않으면 '등급 간 가격 대체로 공정', 단조 감소면 '안전한 등급이 위험 대비 유리'.
"""
import _path  # noqa: F401
from pathlib import Path

SRC = (Path(__file__).parent / '30_portfolio_sharpe.py').read_text()
exec(SRC[:SRC.index("sharpe = lambda")])

sharpe = lambda x: x.mean() / x.std(ddof=1)
TBv = ev['tb'].to_numpy()
R_loan = ((ev['fv'] / ev['funded_amnt']) ** (12 / 36) - 1).to_numpy()
x_loan = R_loan - TBv
npv = (ev['fv'] / (1 + ev['tb']) ** 3 - ev['funded_amnt']).to_numpy()


def sortino(x):
    down = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
    return x.mean() / down if down > 0 else np.nan


def cvar(x, a=0.05):
    return np.sort(x)[: max(1, int(len(x) * a))].mean()


rules = {'기대수익 순': ev['s_er'].to_numpy(), '부도확률 낮은 순': -ev['p'].to_numpy(), '금리 낮은 순': -ev['r'].to_numpy()}
rows = []
for nm, s in list(rules.items()) + [('무작위', None)]:
    for q in QS:
        draws = [top(np.random.default_rng(100 + k).random(len(ev)), q) for k in range(5)] if s is None else [top(s, q)]
        acc = []
        for a in draws:
            xa = x_loan[a]
            acc.append([sharpe(xa), sortino(xa), cvar(xa), npv[a].sum() / 1e6, npv[a].sum() / ev['funded_amnt'].to_numpy()[a].sum()])
        m = np.mean(acc, axis=0)
        rows.append({'전략': nm, '승인비율': q, '건별 Sharpe': m[0], '건별 Sortino': m[1], '하위5% 평균': m[2],
                     '총 NPV(백만$)': m[3], 'NPV/투자1$': m[4]})
T = pd.DataFrame(rows)
T.round(4).to_csv(RESULTS / '33_sortino_npv.csv', index=False)
diff_rank = 0
for q in QS:
    t = T[T['승인비율'] == q].set_index('전략')
    if list(t['건별 Sharpe'].rank()) != list(t['건별 Sortino'].rank()):
        diff_rank += 1

# 등급 중립 롱숏: 발행월마다, 등급마다 신호 상위 30% 롱, 하위 30% 숏
signals = {'기대수익 점수': ev['s_er'].to_numpy(), '부도확률(낮을수록)': -ev['p'].to_numpy(),
           'FICO(높을수록)': ev['fico_range_low'].to_numpy(float), '연소득(높을수록)': ev['annual_inc'].to_numpy(float),
           '부채비율(낮을수록)': -ev['dti'].to_numpy(float),
           '최근 24개월 개설 계좌(적을수록)': -ev['acc_open_past_24mths'].to_numpy(float),
           '무작위(위약)': np.random.default_rng(7).random(len(ev))}
grade = ev['grade'].astype(str).to_numpy()
ym = ev['ym'].to_numpy()
cell = pd.Series(range(len(ev))).groupby([ym, grade]).indices


def legs(sig, frac=0.3):
    L = np.zeros(len(ev), bool)
    S = np.zeros(len(ev), bool)
    for _, ix in cell.items():
        s = sig[ix]
        ok = ~np.isnan(s)
        if ok.sum() < 10:
            continue
        lo, hi = np.nanquantile(s, frac), np.nanquantile(s, 1 - frac)
        L[ix[ok & (s >= hi)]] = True
        S[ix[ok & (s <= lo)]] = True
    return L, S


ls_rows, ls_series = [], {}
allm = np.array([returns.pooled_irr(ev.iloc[grp[m]], fee=FEE) for m in months])
for nm, sig in signals.items():
    L, S = legs(sig)
    rl = np.array([returns.pooled_irr(ev.iloc[grp[m][L[grp[m]]]], fee=FEE) for m in months])
    rs = np.array([returns.pooled_irr(ev.iloc[grp[m][S[grp[m]]]], fee=FEE) for m in months])
    sp = rl - rs
    ls_series[nm] = sp
    ls_rows.append({'신호': nm, '롱 - 숏(%p)': 100 * sp.mean(), '양수 달 비율': (sp > 0).mean(), '스프레드 Sharpe': sharpe(sp),
                    '롱 - 전체(롱온리, %p)': 100 * (rl - allm).mean()})
LS = pd.DataFrame(ls_rows)
n = len(months)
rng = np.random.default_rng(2026)
starts = rng.integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
idx = np.concatenate([np.arange(BLOCK)[None, :] + starts[:, [j]] for j in range(starts.shape[1])], axis=1)[:, :n]
lo_, hi_ = [], []
for nm in LS['신호']:
    bs = ls_series[nm][idx].mean(axis=1)
    lo_.append(100 * np.quantile(bs, 0.05))
    hi_.append(100 * np.quantile(bs, 0.95))
LS['90% 하한'], LS['90% 상한'] = lo_, hi_
LS['판정'] = np.where((LS['90% 하한'] > 0) & (LS['양수 달 비율'] >= 0.75), '신호 있음', '신호 불확실')
LS.round(4).to_csv(RESULTS / '33_longshort.csv', index=False)

# 등급 간: 등급별 발행월 초과수익(IRR - 국채)과 그 Sharpe
g_rows = []
for g in sorted(set(grade)):
    ex = []
    for m in months:
        ix = grp[m][grade[grp[m]] == g]
        if len(ix) >= 20:
            ex.append(returns.pooled_irr(ev.iloc[ix], fee=FEE) - TBv[ix].mean())
    ex = np.array(ex)
    g_rows.append({'등급': g, '발행월 수': len(ex), '평균 초과수익(%p)': 100 * ex.mean(), '표준편차(%p)': 100 * ex.std(ddof=1),
                   'Sharpe': sharpe(ex), '국채 이하 달': int((ex <= 0).sum())})
G = pd.DataFrame(g_rows)
G.round(4).to_csv(RESULTS / '33_grade_sharpe.csv', index=False)

pd.set_option('display.width', 240)
for col in ['건별 Sharpe', '건별 Sortino', '하위5% 평균', '총 NPV(백만$)', 'NPV/투자1$']:
    print(f'\n== {col}')
    print(T.pivot(index='승인비율', columns='전략', values=col).round(3).to_string())
print(f'\nSortino 순위가 Sharpe 순위와 다른 승인비율: {diff_rank}/9 ->', '추가 정보 있음' if diff_rank >= 3 else 'Sharpe와 같은 정보')
print('\n== 등급 중립 롱숏(상위 30% - 하위 30%, 발행월 묶음 IRR)')
print(LS.round(3).to_string(index=False))
print('\n== 등급별 발행월 초과수익')
print(G.round(3).to_string(index=False))
sh = G.set_index('등급')['Sharpe']
mono = sh.is_monotonic_decreasing
print('등급 간 판정:', '안전한 등급이 위험 대비 유리(단조 감소)' if mono else '단조 아님 -> 등급 간 가격 대체로 공정')
print(f'총 {(time.time() - t0) / 60:.1f}분')
