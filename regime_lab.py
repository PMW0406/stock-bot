# -*- coding: utf-8 -*-
"""
국면(레짐) 신호 실험실 — v14 신고가 신호에 어떤 게이트가 최적인가.
 lab_cache(1187종목 5.8년)에서 전 기간 v14 신호 생성(게이트 없이) 후,
 국면 정의별로 필터해 트레이드/포트폴리오 성과 비교.
 청산: 15거래일 or 종가 -10% 손절 / 진입: 익일 시가(갭 ±필터).
"""
import pickle, numpy as np, pandas as pd, sys, types
class _P:
    def __getattr__(s,k): return _p
    def __setattr__(s,k,v): pass
    def __call__(s,*a,**k): return _p
_p=_P()
pl=types.ModuleType("plotly"); pl.__path__=[]; pl.__getattr__=lambda k:_p; sys.modules["plotly"]=pl
for s in ("io","graph_objects","express","subplots","offline","figure_factory"):
    m=types.ModuleType("plotly."+s); m.__getattr__=lambda k:_p; setattr(pl,s,m); sys.modules["plotly."+s]=m
import FinanceDataReader as fdr
import warnings; warnings.filterwarnings("ignore")
COST=0.3; HOLD=15; STOP=0.10

D=pickle.load(open("lab_cache.pkl","rb"))
tdays=D["tdays"]; data=D["data"]; dpm={d:i for i,d in enumerate(tdays)}
ndays=len(tdays); years=sorted(set(d[:4] for d in tdays)); yr_of=[d[:4] for d in tdays]

# ── 종목별 신호 & 브레드스 집계 ──
above200=np.zeros(ndays); total200=np.zeros(ndays)
nearhi=np.zeros(ndays); nearlo=np.zeros(ndays)
trades=[]   # (sig_dayidx, exit_dayidx, ret%)
for tk,v in data.items():
    C=v["C"].astype(np.float64); O=v["O"].astype(np.float64)
    H=v["H"].astype(np.float64); L=v["L"].astype(np.float64); V=v["V"]
    n=len(C)
    if n<300: continue
    s=pd.Series(C)
    ma5=s.rolling(5).mean().values; ma20=s.rolling(20).mean().values
    ma200=s.rolling(200).mean().values
    h52=pd.Series(H).rolling(252).max().values
    l52=pd.Series(L).rolling(252).min().values
    aval=pd.Series(C*V).rolling(20).mean().values
    dates=v["dates"]
    for j in range(252,n-1):
        di=dpm.get(dates[j])
        if di is None: continue
        # 브레드스 집계
        if not np.isnan(ma200[j]):
            total200[di]+=1
            if C[j]>ma200[j]: above200[di]+=1
        if h52[j]>0 and C[j]/h52[j]-1>=-0.05: nearhi[di]+=1
        if l52[j]>0 and C[j]/l52[j]-1<=0.05: nearlo[di]+=1
        # v14 신호 (게이트 없음)
        if (h52[j]>0 and C[j]/h52[j]-1>=-0.05 and aval[j]>=3e9
            and not np.isnan(ma20[j]) and ma5[j]>ma20[j]):
            buy=O[j+1]
            if buy<=0: continue
            gap=(buy-C[j])/C[j]*100
            if gap>=2.0 or gap<=-3.0: continue
            end=min(j+1+HOLD,n-1); r=None; off=end-j-1
            sp=buy*(1-STOP)
            for k in range(j+1,end+1):
                if C[k]<=sp: r=C[k]/buy-1; off=k-j-1; break
            if r is None: r=C[end]/buy-1
            trades.append((di, di+2+off, r*100-COST))
print(f"전 기간 v14 신호(게이트 없음): {len(trades)}건")

breadth = np.where(total200>0, above200/np.maximum(total200,1), np.nan)
net_nh  = nearhi-nearlo

# ── 지수 기반 게이트 ──
ks=fdr.DataReader("KS11",tdays[0],tdays[-1])
ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
ksc=ks["Close"].reindex(tdays).ffill()
ma120=ksc.rolling(120).mean()
r1=(ksc>ma120).fillna(False).values
vol20=(ksc.pct_change().rolling(20).std()*np.sqrt(252)).values
# 히스테리시스: 5일 연속 이탈시 OFF, 5일 연속 회복시 ON
r4=np.zeros(ndays,bool); state=False; cnt=0
raw=r1
for t in range(ndays):
    if raw[t]!=state:
        cnt+=1
        if cnt>=5: state=raw[t]; cnt=0
    else: cnt=0
    r4[t]=state

REGIMES={
 "R0 게이트 없음":            np.ones(ndays,bool),
 "R1 코스피>120MA (현행)":    r1,
 "R2 브레드스>50% (200MA위)": breadth>0.50,
 "R2' 브레드스>40%":          breadth>0.40,
 "R3 신고가권 5종목↑ (자기참조)": nearhi>=5,
 "R4 지수 히스테리시스(5일확인)":  r4,
 "R5 지수변동성<25%":          vol20<0.25,
 "R6 신고가-신저가 순증>0":      net_nh>0,
 "R1+R2 (지수&브레드스)":      r1&(breadth>0.50),
 "R1+R3 (지수&신고가수)":      r1&(nearhi>=5),
}

def evaluate(reg):
    tr=[t for t in trades if reg[t[0]]]
    if len(tr)<50: return None
    r=np.array([t[2] for t in tr]); w=r>0
    # 포트폴리오 K12
    by={}
    for e,x,rr in tr: by.setdefault(e,[]).append((x,rr/100))
    cash=1.0; pos=[]; eq=np.empty(ndays)
    for t in range(ndays):
        keep=[]
        for x,sz,rr in pos:
            if x<=t: cash+=sz*(1+rr)
            else: keep.append((x,sz,rr))
        pos=keep; e=cash+sum(s for _,s,_ in pos); eq[t]=e
        for x,rr in by.get(t,[]):
            if len(pos)>=12: break
            size=e/12
            if 0<size<=cash+1e-12: cash-=size; pos.append((x,size,rr))
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()*100
    last={y:max(i for i,yy in enumerate(yr_of) if yy==y) for y in years}
    yret={}; prev=1.0
    for y in years: yret[y]=(eq[last[y]]/prev-1)*100; prev=eq[last[y]]
    cagr=((eq[-1])**(1/(ndays/245))-1)*100
    return len(tr), w.mean()*100, r.mean(), cagr, mdd, yret

print(f"\n{'국면 게이트':26s} | {'신호':>6} {'승률':>5} {'평균':>6} | {'CAGR':>6} {'MDD':>7} | 연도별")
for tag,reg in REGIMES.items():
    res=evaluate(np.nan_to_num(reg,nan=0).astype(bool))
    if res is None: print(f"{tag:26s} | 표본부족"); continue
    n,wr,avg,cagr,mdd,yret=res
    ys=" ".join(f"{y[2:]}:{yret[y]:+.0f}" for y in years)
    print(f"{tag:26s} | {n:6d} {wr:4.1f}% {avg:+5.2f}% | {cagr:+5.1f}% {mdd:6.1f}% | {ys}")
print("\nREGIME_LAB_DONE")
