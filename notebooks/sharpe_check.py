import warnings, time, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
P='/Users/hyun/Desktop/learningClub/data/processed/lc_train.parquet'
RF=0.05
t0=time.time()
raw=pd.read_parquet(P); raw=raw[raw.loan_status.notna()].copy()
raw['ym']=pd.to_datetime(raw.issue_d,format='%b-%Y'); raw['y']=raw.ym.dt.year
raw['r']=pd.to_numeric(raw.int_rate.astype('string').str.strip().str.rstrip('%')).astype(float)/100
raw['T']=raw.term.astype('string').str.extract(r'(\d+)')[0].astype(int)

# --- 수업 샘플 필터 역추적
print('class target rates [6.93,10.57,14.13,18.58,24.41,29.60,30.87] bad [.096,.197,.306,.417,.467,.509,.545] overall .246')
for name,m in {'2017 all resolved':(raw.y==2017),'2017 36m':(raw.y==2017)&(raw['T']==36),'2017 60m':(raw.y==2017)&(raw['T']==60)}.items():
    s=raw[m&raw.loan_status.isin(['Fully Paid','Charged Off','Default'])]; b=s.loan_status.ne('Fully Paid')
    print(name,len(s),round(b.mean(),3),(s.groupby('grade').r.mean()*100).round(2).tolist(),b.groupby(s.grade).mean().round(3).tolist())
s=raw[(raw.y==2017)]; print(s.loan_status.value_counts().to_string())
b=s.loan_status.isin(['Charged Off','Default','Late (31-120 days)','Late (16-30 days)','In Grace Period']); k=s.loan_status.isin(['Fully Paid','Charged Off','Default','Late (31-120 days)','Late (16-30 days)','In Grace Period'])
print('2017 incl late',round(b[k].mean(),3),b[k].groupby(s.grade[k]).mean().round(3).tolist())

# --- 종결 대출, 수익률 정의
d=raw[raw.loan_status.isin(['Fully Paid','Charged Off','Default'])].copy()
d['bad']=d.loan_status.ne('Fully Paid').astype('int8')
lp=pd.to_datetime(d.last_pymnt_d,format='%b-%Y',errors='coerce')
n=((lp.dt.year-d.ym.dt.year)*12+(lp.dt.month-d.ym.dt.month)).fillna(1).clip(1,None)
n=np.minimum(n,d['T']).astype(float)
moic=(d.total_pymnt/d.funded_amnt).astype(float)
d['RA']=np.where(d.bad==0,d.r,0.0)                       # 수업 정의
d['RB']=(moic-1)*12/d['T']                               # 단순 연환산
i=(1+RF)**(1/12)-1                                        # 받은 돈을 rf로 재투자, 만기 시점 부로 연환산
fv=(moic/n)*((1+i)**d['T']-(1+i)**(d['T']-n))/i
d['RC']=np.clip(fv,1e-3,None)**(12/d['T'])-1
i2=(1.02)**(1/12)-1
fv2=(moic/n)*((1+i2)**d['T']-(1+i2)**(d['T']-n))/i2
d['RD']=np.clip(fv2,1e-3,None)**(12/d['T'])-1
RFS={'RA':.05,'RB':.05,'RC':.05,'RD':.02}
print('\n수익률 요약'); print(d[['RA','RB','RC']].describe().round(4).to_string())
print(d.groupby('bad')[['RA','RB','RC']].mean().round(4).to_string())
print(d.groupby('grade')[['RA','RB','RC']].mean().round(4).to_string())

# --- 특성
for c in ['earliest_cr_line','sec_app_earliest_cr_line']:
    d[c]=pd.to_datetime(d[c],format='%b-%Y',errors='coerce').dt.year.astype('float32')
d['revol_util']=pd.to_numeric(d.revol_util.astype('string').str.rstrip('%'),errors='coerce')
EX=['id','loan_status','issue_d','int_rate','term','url','emp_title','title','zip_code','policy_code','pymnt_plan','next_pymnt_d','last_pymnt_d','last_credit_pull_d','last_fico_range_high','last_fico_range_low','debt_settlement_flag','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt','out_prncp','out_prncp_inv','deferral_term','payment_plan_start_date','orig_projected_additional_accrued_interest','ym','y','bad','RA','RB','RC','RD']
EX+=[c for c in d.columns if c.startswith(('hardship_','settlement_'))]
F=[c for c in d.columns if c not in EX]
for c in F:
    if not pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].astype('category')
print('특성',len(F),'r' in F,'T' in F,flush=True)

def sharpe(e,a):
    x=np.where(a,e,0.0); s=x.std(ddof=1); return x.mean()/s if s>0 else 0.0
def fit(Xtr,ytr,Xva,yva,obj):
    m=lgb.LGBMClassifier if obj=='bin' else lgb.LGBMRegressor
    k=dict(n_estimators=1500,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,min_child_samples=200,reg_lambda=5,cat_smooth=50,n_jobs=-1,verbose=-1)
    mdl=m(**k).fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
    return mdl
def pick(score,e,grid=np.linspace(0,0.98,99)):
    # score 높을수록 승인. val에서 Sharpe 최대인 승인 비율의 임계값
    best=(-9,None)
    for q in grid:
        t=np.quantile(score,q); v=sharpe(e,score>=t)
        if v>best[0]: best=(v,t)
    return best[1],best[0]

def experiment(tag,tr,va,te,extra=[]):
    feats=F+extra
    out=[]
    Xtr,Xva,Xte=tr[feats],va[feats],te[feats]
    t=time.time()
    pdm=fit(Xtr,tr.bad,Xva,va.bad,'bin'); pv=pdm.predict_proba(Xva)[:,1]; pt=pdm.predict_proba(Xte)[:,1]
    print(tag,'PD AUC',round(roc_auc_score(te.bad,pt),4),'trees',pdm.best_iteration_,f'{time.time()-t:.0f}s',flush=True)
    # 수업식 11변수 로지스틱
    base=['loan_amnt','r','fico_range_low','fico_range_high','dti','annual_inc','open_acc','total_acc','inq_last_6mths']
    def lx(x):
        z=pd.concat([x[base].astype(float),pd.get_dummies(x.grade.astype(str),drop_first=True).astype(float)],axis=1)
        return z.fillna(z.median())
    sc=StandardScaler().fit(lx(tr)); lg=LogisticRegression(max_iter=300).fit(sc.transform(lx(tr)),tr.bad)
    lv=lg.predict_proba(sc.transform(lx(va)))[:,1]; lt=lg.predict_proba(sc.transform(lx(te)))[:,1]
    for R in ['RA','RB','RC','RD']:
        rf=RFS[R]; ev,et,etr=(va[R]-rf).values,(te[R]-rf).values,(tr[R]-rf).values
        row={'split':tag,'R':R,'전부승인':sharpe(et,np.ones(len(et),bool))}
        strategies={}
        strategies['logit PD컷']=(-lv,-lt)
        strategies['LGBM PD컷']=(-pv,-pt)
        strategies['금리순']=(-va.r.values,-te.r.values)
        mu=fit(Xtr,etr,Xva,ev,'reg'); m2=fit(Xtr,etr**2,Xva,ev**2,'reg')
        muv,mut=mu.predict(Xva),mu.predict(Xte); m2v,m2t=np.clip(m2.predict(Xva),1e-6,None),np.clip(m2.predict(Xte),1e-6,None)
        strategies['E[e]컷']=(muv,mut)
        strategies['E[e]/E[e²]컷']=(muv/m2v,mut/m2t)
        if R=='RA':
            r_v,r_t=va.r.values,te.r.values
            f=lambda p,r:((1-p)*(r-RF)-p*RF)/((1-p)*(r-RF)**2+p*RF**2)
            strategies['PD→E/E²(해석식)']=(f(pv,r_v),f(pt,r_t))
        sel={}
        for nme,(sv,st) in strategies.items():
            th,vb=pick(sv,ev); a=st>=th; sel[nme]=a
            row[nme]=sharpe(et,a); row[nme+'_승인%']=a.mean()*100
            row[nme+'_val']=vb
        def coh(a):
            z=pd.DataFrame({'e':et,'a':a,'ym':te.ym.values,'y':te.y.values}); z=z[z.y.between(2012,2016)]
            m=z.assign(x=np.where(z.a,z.e,0.0)).groupby('ym').x.mean()
            return m.mean()/m.std(ddof=1)*np.sqrt(12) if len(m)>6 else np.nan
        row['코호트_전부승인']=coh(np.ones(len(et),bool)); row['코호트_PD컷']=coh(sel['LGBM PD컷']); row['코호트_E/E2']=coh(sel['E[e]/E[e²]컷'])
        # bootstrap: 최종 규칙 vs 전부승인 / vs PD컷
        rng=np.random.default_rng(0); key='E[e]/E[e²]컷'; dif=[];dif2=[]
        N=len(et)
        for _ in range(200):
            ix=rng.integers(0,N,N); e=et[ix]
            dif.append(sharpe(e,sel[key][ix])-sharpe(e,np.ones(N,bool)))
            dif2.append(sharpe(e,sel[key][ix])-sharpe(e,sel['LGBM PD컷'][ix]))
        row['Δ전부승인_CI']=f'{np.percentile(dif,2.5):+.3f}~{np.percentile(dif,97.5):+.3f}'
        row['ΔPD컷_CI']=f'{np.percentile(dif2,2.5):+.3f}~{np.percentile(dif2,97.5):+.3f}'
        out.append(row)
        print({k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()},flush=True)
    return out

res=[]
rng=np.random.default_rng(42); u=rng.random(len(d))
tr,va,te=d[u<.6],d[(u>=.6)&(u<.8)],d[u>=.8]
res+=experiment('random',tr,va,te)
d['iy']=d.y.astype('float32')
tr,va,te=d[u<.6],d[(u>=.6)&(u<.8)],d[u>=.8]
res+=experiment('random+발행연도',tr,va,te,['iy'])
tr,va,te=d[d.y<=2014],d[d.y==2015],d[d.y==2016]
res+=experiment('time(≤14/15/16)',tr,va,te)
pd.DataFrame(res).to_csv(sys.argv[1],index=False)
print(f'총 {(time.time()-t0)/60:.1f}분')
