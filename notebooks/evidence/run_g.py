"""G: 우승 논문 규칙과 비교. 같은 표본·같은 국채금리·같은 수익률에서 승인 규칙만 바꾼다. + 역U자 분해."""
import warnings, time, os, re
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
src=open('run_all.py').read(); exec(src[:src.index('# ---------- A: 경쟁위험')])  # 준비/함수 재사용
OUT='results'
g=[]
for s,T in [('GS3',36),('GS5',60)]:
    x=pd.read_csv('../../data/external/%s.csv'%s); x.columns=['ym','tb']; x['ym']=pd.to_datetime(x.ym); x['tb']=pd.to_numeric(x.tb,errors='coerce')/100; x['Tk']=T; g.append(x)
g=pd.concat(g); d['Tk']=d.Tm.astype(int); d=d.merge(g,on=['ym','Tk'],how='left'); d.index=range(len(d))
hpr=(d.total_pymnt/d.loan_amnt).astype(float); d['RP']=np.clip(hpr,1e-6,None)**(12/d.Tk)-1     # 논문 수익률
n_=np.minimum(d.tau,d.Tk).astype(float); mo=(d.total_pymnt/d.funded_amnt).astype(float); i_=(1+d.tb)**(1/12)-1
fv=(mo/n_)*((1+i_)**d.Tk-(1+i_)**(d.Tk-n_))/i_; d['RM']=np.clip(fv,1e-3,None)**(12/d.Tk)-1          # 받은 돈을 국채로 재투자
d['iy']=d.y.astype('float32'); mth=d.ym.dt.month; d['ms']=np.sin(2*np.pi*mth/12).astype('float32'); d['mc']=np.cos(2*np.pi*mth/12).astype('float32')
FY=F+['iy','ms','mc']
def paper_rule(ptr,rtr,btr,pv,pe,J=10):
    edges=np.quantile(ptr,np.linspace(0,1,11)); edges[0],edges[-1]=-1,2
    dec=lambda p:np.clip(np.searchsorted(edges,p,side='right')-1,0,9)
    dt=dec(ptr); rn=np.array([rtr[(dt==k)&(btr==0)].mean() for k in range(10)]); rd=np.array([rtr[(dt==k)&(btr==1)].mean() for k in range(10)])
    ER=lambda p:(1-p)*rn[dec(p)]+p*rd[dec(p)]
    et=ER(ptr); b=np.quantile(et,np.linspace(0,1,J+1)); b[0],b[-1]=-9,9
    sig=np.array([rtr[(et>=b[j])&(et<b[j+1])].std() for j in range(J)]); S=lambda e:sig[np.clip(np.searchsorted(b,e,side='right')-1,0,J-1)]
    return ER(pv),S(ER(pv)),ER(pe),S(ER(pe))
def met(te,a,R):
    e=(te[R]-te.tb).values; x=np.where(a,e,0.); m=e[a]
    return dict(승인비율=a.mean(),샤프=sharpe(e,a),승인평균초과=m.mean() if a.any() else np.nan,승인표준편차=m.std() if a.any() else np.nan,승인부도율=te.bad.values[a].mean() if a.any() else np.nan)
def one(tag,seed,tr,va,te,feats):
    Xt,Xv,Xe=tr[feats],va[feats],te[feats]; pm=fit(Xt,tr.bad,Xv,va.bad,'bin')
    ptr,pv,pe=pm.predict_proba(Xt)[:,1],pm.predict_proba(Xv)[:,1],pm.predict_proba(Xe)[:,1]
    out=[]
    for R in ['RP','RM']:
        rtr=tr[R].values; ev=(va[R]-va.tb).values; et=(tr[R]-tr.tb).values
        Ev,Sv,Ee,Se=paper_rule(ptr,rtr,tr.bad.values,pv,pe)
        tv,te_=va.tb.values,te.tb.values
        mu=fit(Xt,et,Xv,ev,'reg'); m2=fit(Xt,et**2,Xv,ev**2,'reg')
        a1,a2=mu.predict(Xv),mu.predict(Xe); b1,b2=np.clip(m2.predict(Xv),1e-6,None),np.clip(m2.predict(Xe),1e-6,None)
        sd1,sd2=np.sqrt(np.clip(b1-a1**2,1e-6,None)),np.sqrt(np.clip(b2-a2**2,1e-6,None))
        S={'PD컷':(-pv,-pe),'논문 십분위 (E-rf)/σ':((Ev-tv)/Sv,(Ee-te_)/Se),'십분위 (E-rf)/σ²':((Ev-tv)/Sv**2,(Ee-te_)/Se**2),'십분위 E만':(Ev-tv,Ee-te_),
           'LGBM μ':(a1,a2),'LGBM μ/σ':(a1/sd1,a2/sd2),'LGBM μ/E[e²]':(a1/b1,a2/b2)}
        out.append(dict(분할=tag,seed=seed,R=R,전략='전부승인',AUC=roc_auc_score(te.bad,pe),**met(te,np.ones(len(te),bool),R)))
        for nm,(sv,s2) in S.items():
            a=s2>=pick(sv,ev); out.append(dict(분할=tag,seed=seed,R=R,전략=nm,AUC=roc_auc_score(te.bad,pe),**met(te,a,R)))
    return out
f='results/G_compare.csv'
for seed in range(10):
    u=np.random.default_rng(2000+seed).random(len(d)); tr,va,te=d[u<.6],d[(u>=.6)&(u<.8)],d[u>=.8]
    r=one('무작위+발행연도',seed,tr,va,te,FY)+one('무작위',seed,tr,va,te,F)
    pd.DataFrame(r).round(5).to_csv(f,mode='a',header=not os.path.exists(f),index=False); log('G seed',seed)
for Y in [2014,2015,2016,2017]:
    r=one('시간',Y,d[d.y<=Y-2],d[d.y==Y-1],d[d.y==Y],F)
    pd.DataFrame(r).round(5).to_csv(f,mode='a',header=False,index=False); log('G time',Y)
# 역U자 분해: SR = sqrt(a) m / sqrt(s^2+(1-a)m^2), 표본 수 N과 무관한지
u=np.random.default_rng(1).random(len(d)); tr,va,te=d[u<.6],d[(u>=.6)&(u<.8)],d[u>=.8]
pm=fit(tr[F],tr.bad,va[F],va.bad,'bin'); pe=pm.predict_proba(te[F])[:,1]; e=(te.RP-te.tb).values
rows=[]
rng=np.random.default_rng(0)
for sh in [.02,.05,.1,.2,.27,.4,.6,.8,1.]:
    a=pe<=np.quantile(pe,sh); m,s=e[a].mean(),e[a].std(); rows.append(dict(승인비율=a.mean(),샤프=sharpe(e,a),공식=np.sqrt(a.mean())*m/np.sqrt(s**2+(1-a.mean())*m**2),승인대출만_m_over_s=m/s,승인평균=m,승인표준편차=s))
pd.DataFrame(rows).round(4).to_csv('results/G_invU.csv',index=False)
a=pe<=np.quantile(pe,.27); rows=[]
for N in [100,1000,10000,len(te)]:
    v=[];pm_=[]
    for _ in range(300):
        ix=rng.choice(len(te),N,replace=False) if N<len(te) else rng.integers(0,N,N); v.append(sharpe(e[ix],a[ix])); pm_.append(np.where(a[ix],e[ix],0).mean())
    rows.append(dict(대출수N=N,샤프평균=np.mean(v),샤프표준편차=np.std(v),포트폴리오평균수익_표준편차=np.std(pm_)))
pd.DataFrame(rows).round(5).to_csv('results/G_N.csv',index=False)
log('G 완료')
