import pandas as pd, numpy as np
from scipy.optimize import brentq
B='/Users/hyun/Desktop/learningClub/'
g=[]
for s,T in [('GS3',36),('GS5',60)]:
    x=pd.read_csv(B+f'data/external/{s}.csv'); x.columns=['ym','tb']; x['ym']=pd.to_datetime(x.ym); x['tb']=pd.to_numeric(x.tb,errors='coerce')/100; x['T']=T; g.append(x)
g=pd.concat(g)
c=['loan_status','issue_d','last_pymnt_d','term','funded_amnt','loan_amnt','total_pymnt','int_rate']
d=pd.read_parquet(B+'data/processed/lc_train.parquet',columns=c)
d=d[d.loan_status.isin(['Fully Paid','Charged Off','Default'])].copy()
d['ym']=pd.to_datetime(d.issue_d,format='%b-%Y'); d['T']=d.term.str.extract(r'(\d+)')[0].astype(int); d['y']=d.ym.dt.year
d=d.merge(g,on=['ym','T'],how='left'); d['bad']=d.loan_status.ne('Fully Paid')
d['r']=pd.to_numeric(d.int_rate.str.strip().str.rstrip('%'))/100
hpr=d.total_pymnt/d.loan_amnt-1; d['rp']=(1+hpr).clip(lower=1e-6)**(12/d['T'])-1
print('논문 재현: 정상 평균/중앙',d[~d.bad].rp.mean().round(4),d[~d.bad].rp.median().round(4),'부도',d[d.bad].rp.mean().round(4),d[d.bad].rp.median().round(4))
print('전체 평균수익',round(d.rp.mean(),4),'변동성',round(d.rp.std(),4),'국채',round(d.tb.mean(),4))
print('전부승인 SR',round((d.rp.mean()-d.tb.mean())/d.rp.std(),4), '| 초과수익 기준',round((d.rp-d.tb).mean()/(d.rp-d.tb).std(),4))
# 만기까지 정상 상환해도 이 식이 주는 값
i=d.r/12; A=i/(1-(1+i)**(-d['T'])); full=(A*d['T'])**(12/d['T'])-1
print('약정 금리 평균',round(d.r.mean(),4),'약정대로 전액 상환 시 논문식 수익',round(full.mean(),4))
# 코호트 통합 IRR
lp=pd.to_datetime(d.last_pymnt_d,format='%b-%Y',errors='coerce')
n=((lp.dt.year-d.ym.dt.year)*12+lp.dt.month-d.ym.dt.month).fillna(1).clip(1,None)
d['n']=np.minimum(n,d['T']+24).astype(int)
d=d.merge(pd.read_parquet(B+'data/processed/lc_train.parquet',columns=['id','installment','issue_d','funded_amnt','total_pymnt','loan_status']).drop_duplicates(['issue_d','funded_amnt','total_pymnt','loan_status','id']).groupby(['issue_d','funded_amnt','total_pymnt','loan_status'],as_index=False).installment.first(),on=['issue_d','funded_amnt','total_pymnt','loan_status'],how='left')
def irr(s,fee=0.0):
    H=100; cf=np.zeros(H); cf[0]=-s.funded_amnt.sum()
    inst=s.installment.values; tot=s.total_pymnt.values; nn=s.n.values
    k=np.minimum(nn,np.floor(tot/inst)).astype(int); rem=tot-k*inst
    add=np.zeros(H+2); np.add.at(add,1,inst.sum()); np.add.at(add,k+1,-inst)
    cf[1:]=np.cumsum(add)[1:H]
    np.add.at(cf,np.maximum(nn,k),rem)   # 조기상환 일시금과 회수금은 마지막 시점에
    cf[1:]*=(1-fee)
    f=lambda m:(cf/(1+m)**np.arange(H)).sum()
    return (1+brentq(f,-.08,.08))**12-1
rows=[]
for y,s in d[d.y.between(2010,2018)].groupby('y'):
    rows.append(dict(발행연도=y,건수=len(s),부도율=s.bad.mean(),논문식평균=s.rp.mean(),통합IRR=irr(s),수수료1pct후=irr(s,.01),국채=s.tb.mean()))
t=pd.DataFrame(rows).round(4); print(t.to_string(index=False)); t.to_csv('results/paper_irr.csv',index=False)
s=d[d.y.between(2012,2014)]; print('2012~14 (거의 전부 종결) IRR',round(irr(s),4),'수수료후',round(irr(s,.01),4),'국채',round(s.tb.mean(),4),'논문식',round(s.rp.mean(),4),'논문식SR',round((s.rp-s.tb).mean()/(s.rp-s.tb).std(),3))
s=d[d.y.between(2012,2016)]; print('2012~16 IRR',round(irr(s),4),'수수료후',round(irr(s,.01),4),'국채',round(s.tb.mean(),4),'논문식',round(s.rp.mean(),4))
