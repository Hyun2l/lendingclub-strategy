"""발행월 코호트별 생애 IRR: '그 달에 투자를 시작했다면' 수익의 분포 = 시기 위험."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import brentq
SRC=open('sim_proto.py').read()
exec(SRC[:SRC.index('# 2) 등급별')]); exec(SRC[SRC.index('# 3) 전략 비교'):SRC.index('rows=[];ser={}')])
c36=(raw.Tm.values==36)&raw.y.between(2013,2016).values
coh=s-(2013*12); K=50
def cohort_irr(w):
    m=(w>0)&c36; CF=np.zeros((48,K+1)); cc=coh[m]
    np.add.at(CF,(cc,0),-(w*P)[m])
    for k in range(1,K+1):
        pay=(A*(k<=kf)+rem*(k==n))[m]*w[m]; np.add.at(CF,(cc,np.full(cc.shape,k)),pay)
    out=[]
    for j in range(48):
        f=lambda x:(CF[j]/(1+x)**np.arange(K+1)).sum()
        try: out.append((1+brentq(f,-.2,.2))**12-1)
        except ValueError: out.append(np.nan)
    return np.array(out)
months=pd.period_range('2013-01','2016-12',freq='M'); rf=tb.GS3.reindex(months.to_timestamp()).values
rows=[];allc={}
for nm,w in S.items():
    x=cohort_irr(w); allc[nm]=x; ex=x-rf
    rows.append(dict(전략=nm,평균IRR=np.nanmean(x),코호트간표준편차=np.nanstd(x),최저코호트=np.nanmin(x),초과수익_샤프=np.nanmean(ex)/np.nanstd(ex),국채이하_코호트수=int((ex<0).sum())))
T=pd.DataFrame(rows).round(4); print(T.to_string(index=False)); T.to_csv(f'{OUT}/S_cohort.csv',index=False)
D=pd.DataFrame(allc,index=months.astype(str)); D['국채']=rf; D.round(4).to_csv(f'{OUT}/S_cohort_monthly.csv')
print(D.groupby(months.year).mean().round(4).to_string())
p=D['기대수익 상위 30%']-D['전부']; q=D['PD 하위 30%']-D['전부']; r=D['기대수익 상위 30%']-D['PD 하위 30%']
print('코호트별 차이 (기대수익30-전부) 평균',round(p.mean(),4),'양수 비율',round((p>0).mean(),2),'| (PD30-전부)',round(q.mean(),4),round((q>0).mean(),2),'| (기대30-PD30)',round(r.mean(),4),round((r>0).mean(),2))
