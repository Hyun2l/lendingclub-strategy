"""계좌 시뮬레이션 시제품: 발행 시점 정보로 대출을 고르고, 월별 현금흐름으로 계좌 수익률을 만든다."""
import warnings, time, os, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
src=open('run_all.py').read(); exec(src[:src.index('# ---------- A: 경쟁위험')])
OUT='results'
st=raw.loan_status.astype(str)
cls=np.where(st.str.contains('Fully Paid'),2,np.where(st.str.contains('Charged Off')|st.isin(['Default','Late (31-120 days)','Late (16-30 days)','In Grace Period']),1,0))
s=(raw.ym.dt.year*12+raw.ym.dt.month-1).values
P=raw.funded_amnt.values.astype(float); A=raw.installment.values.astype(float); i=raw.r.values.astype(float)/12
tot=raw.total_pymnt.values.astype(float); n=raw.tau.values.astype(int)
kf=np.minimum(n,np.floor(tot/np.maximum(A,1e-9))).astype(int); rem=np.where(cls>0,tot-kf*A,0.0)
# 1) 복원 점검
paid=cls==2
log('정상상환 중 만기 전 상환(조기상환) 비율',round((n[paid]<raw.Tm.values[paid]-1).mean(),3),
    '| 일시금/총상환 중앙값',round(np.median(rem[paid]/tot[paid]),3),'| 복원 오차(일시금<0) 비율',round((rem<-1).mean(),4))
Bk=lambda k:np.maximum(P*(1+i)**k-A*((1+i)**k-1)/np.where(i>0,i,1e-9),0)
g=[]
for sname,T in [('GS3',36),('GS5',60)]:
    x=pd.read_csv('../../data/external/%s.csv'%sname); x.columns=['ym','tb']; x['ym']=pd.to_datetime(x.ym); x['tb']=pd.to_numeric(x.tb,errors='coerce')/100; g.append(x.set_index('ym').tb.rename(sname))
tb=pd.concat(g,axis=1)
T0=2013*12; T1=2019*12+11
def account(w):
    """w: 대출별 투자 금액 비율(0~1). 월별 (현금, 평가액, 신규투자) -> 월 수익률."""
    H=T1+2; C=np.zeros(H); V=np.zeros(H); I=np.zeros(H)
    m=w>0; ww,ss,PP,nn,kk,cc,rr=w[m],s[m],P[m],n[m],kf[m],cls[m],rem[m]; AA,ii=A[m],i[m]
    ok=ss<=T1; np.add.at(I,ss[ok],ww[ok]*PP[ok]); np.add.at(V,ss[ok],ww[ok]*PP[ok])
    PPm,AAm=PP,AA
    for k in range(1,85):
        t=ss+k; live=(t<=T1)
        if not live.any(): break
        pay=AAm*(k<=kk)+rr*(k==nn)
        Bsched=np.maximum(PPm*(1+ii)**np.minimum(k,kk)-AAm*((1+ii)**np.minimum(k,kk)-1)/np.where(ii>0,ii,1e-9),0)
        bal=np.where((cc>0)&(k>=nn),0.0,Bsched)
        alive=(k<=nn)|(cc==0)
        sel=live&(alive|(k==nn))
        np.add.at(C,t[sel],(ww*pay)[sel]); np.add.at(V,t[sel&alive],(ww*bal)[sel&alive])
    R=(C[1:]+V[1:]-I[1:])/np.where(V[:-1]>0,V[:-1],np.nan)-1
    idx=pd.period_range(f'{(1)}-01','0001-01',freq='M')  # placeholder
    mon=pd.PeriodIndex([pd.Period(year=(t)//12,month=(t)%12+1,freq='M') for t in range(1,H)])
    r=pd.Series(R,index=mon); return r[(r.index>=pd.Period('2013-02','M'))&(r.index<=pd.Period('2019-12','M'))]
def stats(r):
    rfm=tb.GS3.reindex(r.index.to_timestamp()).values/12
    ex=r.values-rfm; cum=(1+r).cumprod(); dd=(cum/cum.cummax()-1).min()
    return dict(연수익=(1+r.mean())**12-1,연변동성=r.std()*np.sqrt(12),샤프_연=ex.mean()/ex.std()*np.sqrt(12),최악월=r.min(),최대낙폭=dd)
# 2) 등급별 금리 vs 실현 (2012~14 발행, 거의 전부 종결)
rows=[]
for gname in sorted(raw.sub_grade.dropna().unique()):
    w=((raw.sub_grade==gname)&raw.y.between(2013,2014)).values.astype(float)
    if w.sum()<2000: continue
    r=account(w); rows.append(dict(세부등급=gname,건수=int(w.sum()),평균금리=raw.r.values[w>0].mean(),부도율=(cls[w>0]==1).mean(),**stats(r)))
SG=pd.DataFrame(rows).round(4); SG.to_csv(f'{OUT}/S_subgrade.csv',index=False); log('세부등급 완료')
# 3) 전략 비교: 발행연도 Y마다 그 해 1월까지의 정보로 학습
F2=[c for c in F]
score_p=np.full(len(raw),np.nan); score_e=np.full(len(raw),np.nan)
for Y in range(2013,2018):
    C_=pd.Timestamp(f'{Y}-01-01'); L=raw[raw.ym<C_]; age=((C_.year-L.ym.dt.year)*12+C_.month-L.ym.dt.month).values
    kn=L.res.values&(L.tau.values<=age); Rr=L[kn]; u=np.random.default_rng(Y).random(len(Rr))<.85
    m=fit(Rr[F2][u],Rr.bad[u],Rr[F2][~u],Rr.bad[~u],'bin')
    lgd=float(1-(Rr.total_pymnt/Rr.funded_amnt)[Rr.bad==1].mean())
    idx=np.where(raw.y.values==Y)[0]; p=m.predict_proba(raw.iloc[idx][F2])[:,1]
    score_p[idx]=-p; score_e[idx]=(1-p)*raw.r.values[idx]-p*lgd
    log('학습',Y,'시점종결',len(Rr),'LGD',round(lgd,3))
inv=raw.y.between(2013,2017).values
def topq(sc,q):
    w=np.zeros(len(raw))
    for Y in range(2013,2018):
        idx=np.where(raw.y.values==Y)[0]; th=np.nanquantile(sc[idx],1-q); w[idx]=(sc[idx]>=th)
    return w
rng=np.random.default_rng(0)
S={'전부':inv.astype(float),'A·B등급만':(inv&raw.grade.isin(['A','B']).values).astype(float),'무작위 30%':(inv&(rng.random(len(raw))<.3)).astype(float),
   'PD 하위 30%':topq(score_p,.3),'PD 하위 50%':topq(score_p,.5),'기대수익 상위 30%':topq(score_e,.3),'기대수익 상위 50%':topq(score_e,.5),
   '기대수익>0 (자연 기준)':(inv&(score_e>0)).astype(float)}
rows=[];ser={}
for nm,w in S.items():
    r=account(w); ser[nm]=r; rows.append(dict(전략=nm,승인비율=w[inv].mean(),**stats(r)))
ST=pd.DataFrame(rows).round(4); ST.to_csv(f'{OUT}/S_strategy.csv',index=False)
pd.DataFrame(ser).to_csv(f'{OUT}/S_monthly.csv')
yr=pd.DataFrame({k:(1+v).groupby(v.index.year).prod()-1 for k,v in ser.items()}).round(4); yr.to_csv(f'{OUT}/S_yearly.csv')
print(SG.to_string(index=False)); print(ST.to_string(index=False)); print(yr.to_string())
log('완료')
