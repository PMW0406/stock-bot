# -*- coding: utf-8 -*-
"""
국면(레짐) 스위치 검증 — 기존 Q2 신호에 '지수 추세필터'를 덧씌워,
강세장에만 트레이딩하고 나머지엔 현금이면 손실을 피하는지 연도별로 확인.
신호 캐시(backtest_multiyear_cache.npz) 재사용 + 지수만 새로 수집(빠름).
"""
import numpy as np, pandas as pd
import backtest_multiyear as M
fdr=M.B.fdr; COST=M.COST; LOOKFWD=M.LOOKFWD
Z=np.load(M.CACHE, allow_pickle=True)
tdays=[d.decode() if isinstance(d,bytes) else str(d) for d in Z["tdays"]]
day=Z["day"]; ret20=Z["ret20"]; score=Z["score"]; lo=Z["lo"]; clo=Z["clo"]; rs80=Z["rs80"]
ndays=len(tdays); dpos={d:i for i,d in enumerate(tdays)}
yr_of=[d[:4] for d in tdays]; ys=sorted(set(yr_of))

# ── 지수 국면 지표 ──
ks=fdr.DataReader("KS11",tdays[0],tdays[-1]); ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
ksc=ks["Close"].reindex(tdays).ffill()
ma200=ksc.rolling(200).mean(); ma60=ksc.rolling(60).mean(); ma120=ksc.rolling(120).mean()
above200=(ksc>ma200).fillna(False).values
above60 =(ksc>ma60).fillna(False).values
rise200 =(ma200.values - np.roll(ma200.values,20) > 0)
rise200[:220]=False
above120=(ksc>ma120).fillna(False).values
kret=ksc.pct_change().fillna(0).values   # 지수 일간수익 (벤치마크용)

regimes={
 "R0 (필터없음=기존)": np.ones(ndays,bool),
 "R1 지수>200MA":      above200,
 "R2 지수>200MA&상승":  above200 & rise200,
 "R3 지수>120MA":      above120,
 "R4 지수>60MA":       above60,
}

def outcomes(regime, stop=None, hold=10):
    fin = (ret20>=rs80[day]) & regime[day]
    out=[]
    for i in np.where(fin)[0]:
        L=lo[i,:hold]; C=clo[i,:hold]; v=~np.isnan(C)
        if not v.any(): continue
        if stop is not None:
            sp=1-stop; hit=np.where(v&(L<=sp))[0]
            if len(hit): off=int(hit[0]); r=-stop
            else: last=int(np.where(v)[0][-1]); off=last; r=float(C[last]-1)
        else:
            last=int(np.where(v)[0][-1]); off=last; r=float(C[last]-1)
        d=int(day[i]); out.append((d+1,d+2+off,r-COST,float(score[i])))
    return out

def port(trades,K=8):
    by={}
    for e,x,r,s in trades: by.setdefault(e,[]).append((s,x,r))
    cash=1.0; pos=[]; eq=np.empty(ndays)
    for t in range(ndays):
        keep=[]
        for x,sz,r in pos:
            if x==t: cash+=sz*(1+r)
            else: keep.append((x,sz,r))
        pos=keep; e=cash+sum(s for _,s,_ in pos); eq[t]=e
        for s,x,r in sorted(by.get(t,[]),key=lambda z:z[0],reverse=True):
            if len(pos)>=K: break
            size=e/K
            if 0<size<=cash+1e-12: cash-=size; pos.append((x,size,r))
    # 미청산 정리
    fin=cash+sum(s for _,s,_ in pos)
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()*100
    # 연도별
    last={y:max(i for i,yy in enumerate(yr_of) if yy==y) for y in ys}
    yret={}; prev=1.0
    for y in ys:
        e_end=eq[last[y]]; yret[y]=(e_end/prev-1)*100; prev=e_end
    tot=eq[last[ys[-1]]]-1
    return tot*100, mdd, yret, eq

# 벤치마크: 지수 사서 국면ON일때만 보유(else 현금)
def bench(regime):
    eq=np.empty(ndays); v=1.0
    for t in range(ndays):
        if t>0 and regime[t-1]: v*= (1+kret[t])
        eq[t]=v
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()*100
    last={y:max(i for i,yy in enumerate(yr_of) if yy==y) for y in ys}
    yret={}; prev=1.0
    for y in ys:
        yret[y]=(eq[last[y]]/prev-1)*100; prev=eq[last[y]]
    return (eq[last[ys[-1]]]-1)*100, mdd, yret

print(f"기간 {tdays[0]}~{tdays[-1]} / 연도 {ys}\n")
print("="*104)
print(" Q2 + 승자태우기(무손절)/10일/K8 에 '지수 국면필터' 덧씌움 — 연도별 수익률")
print("="*104)
print(f"  {'국면필터':22s} | {'5년총':>7} {'CAGR':>6} {'MDD':>6} | " + " ".join(f"{y[2:]:>6}" for y in ys))
for name,reg in regimes.items():
    tot,mdd,yret,_=port(outcomes(reg,None,10))
    cagr=((1+tot/100)**(1/(ndays/245))-1)*100
    print(f"  {name:22s} | {tot:+6.1f}% {cagr:+5.1f}% {mdd:5.1f}% | " + " ".join(f"{yret[y]:+6.1f}" for y in ys))

print("\n [참고] 지수 자체를 국면ON일때만 보유(else현금) 벤치마크")
print(f"  {'국면필터':22s} | {'5년총':>7} {'MDD':>6} | " + " ".join(f"{y[2:]:>6}" for y in ys))
for name in ("R1 지수>200MA","R2 지수>200MA&상승","R4 지수>60MA"):
    tot,mdd,yret=bench(regimes[name])
    print(f"  {name:22s} | {tot:+6.1f}% {mdd:5.1f}% | " + " ".join(f"{yret[y]:+6.1f}" for y in ys))
# 그냥 지수 buy&hold
tot,mdd,yret=bench(np.ones(ndays,bool))
print(f"  {'지수 매수후보유':22s} | {tot:+6.1f}% {mdd:5.1f}% | " + " ".join(f"{yret[y]:+6.1f}" for y in ys))
print("="*104)
print("완료 / 총 done")
