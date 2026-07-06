# -*- coding: utf-8 -*-
"""
전략 리서치용 범용 캐시 — 전 종목/전 거래일에 대해 풍부한 피처 + 앞으로 25일 경로 저장.
한 번 수집(~15분)해두면 서로 다른 전략(추세/역추세/과매도반등 등)을 즉시 비교 가능.
저장: research_cache.npz
"""
import sys, os, time
import backtest_relax as B          # plotly 스텁 + fdr + market_states 재사용
import numpy as np, pandas as pd

fdr=B.fdr
START=B.START_DATE; STARTS=B.START_DATE_STOCK; END=B.END_DATE
MCMIN=B.MARKET_CAP_MIN; MCMAX=B.MARKET_CAP_MAX
LOOKFWD=25
CACHE="research_cache.npz"

def rsi(close, p):
    d=close.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ru=up.ewm(alpha=1/p, adjust=False).mean(); rd=dn.ewm(alpha=1/p, adjust=False).mean()
    rs=ru/rd.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)

def build():
    t0=time.time(); print("시장 지수..."); sys.stdout.flush()
    kospi=fdr.DataReader("KS11",START,END); kosdaq=fdr.DataReader("KQ11",START,END)
    kospi.index=pd.to_datetime(kospi.index).strftime("%Y-%m-%d"); kosdaq.index=pd.to_datetime(kosdaq.index).strftime("%Y-%m-%d")
    tdays=sorted(kospi.index.tolist()); dpos_m={d:i for i,d in enumerate(tdays)}
    mk_s,mk_m=B.market_states(tdays,kospi,kosdaq)
    print(f"시장 strict={int(mk_s.sum())} mild={int(mk_m.sum())}/{len(tdays)}")
    alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
    filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
    tickers=filt["Code"].tolist(); print(f"종목 {len(tickers)}개, 피처+25일경로 수집..."); sys.stdout.flush()
    prev_of={bd:(tdays[i-1] if i>0 else None) for i,bd in enumerate(tdays)}
    valid_buy=tdays[:-1]
    F=dict(day=[],gap=[],
        c_ma20=[],c_ma60=[],ma20_60=[],ma5_20=[],up120=[],
        rsi2=[],rsi14=[],ret1=[],ret5=[],ret20=[],
        dist_hi20=[],dist_lo20=[],down3=[],volr=[],avgval=[],atrp=[],
        cl=[],pb7=[])
    hi=[]; lo=[]; clo=[]; rs_by_day={}; n=0
    for ii,tk in enumerate(tickers):
        if ii%300==0: print(f"  {ii}/{len(tickers)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,STARTS,END)
            if df.empty or len(df)<130: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        n+=1
        C=df["Close"]; Hh=df["High"]; Ll=df["Low"]; Oo=df["Open"]; Vv=df["Volume"]
        ma5=C.rolling(5).mean(); ma20=C.rolling(20).mean(); ma60=C.rolling(60).mean(); ma120=C.rolling(120).mean()
        r2=rsi(C,2); r14=rsi(C,14)
        prevc=C.shift(1); tr=np.maximum(Hh-Ll, np.maximum((Hh-prevc).abs(),(Ll-prevc).abs()))
        atr=tr.ewm(alpha=1/14,adjust=False).mean(); atrp=atr/C*100
        hi20=Hh.rolling(20).max(); lo20=Ll.rolling(20).min()
        down3=(C.diff()<0).astype(int).rolling(3).sum()
        avgval=(Vv*C).rolling(20).mean()
        vol5=Vv.rolling(5).mean()
        # numpy 배열화
        cA=C.values; ma5A=ma5.values; ma20A=ma20.values; ma60A=ma60.values; ma120A=ma120.values
        r2A=r2.values; r14A=r14.values; atrpA=atrp.values
        hi20A=hi20.values; lo20A=lo20.values; down3A=down3.values; avgvA=avgval.values
        HA=Hh.values; LA=Ll.values; VA=Vv.values; vol5A=vol5.values; OA=Oo.values
        dates=list(df.index); dp={d:j for j,d in enumerate(dates)}
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dp: continue
            r=dp[pv]
            if r<120: continue
            m20=ma20A[r]; m60=ma60A[r]; m120=ma120A[r]; m5=ma5A[r]; c=cA[r]
            if np.isnan(m120) or np.isnan(m60) or m20<=0 or m60<=0 or m5<=0: continue
            ret20=(c-cA[r-20])/cA[r-20]*100
            rs_by_day.setdefault(pv,{})[tk]=ret20
            if bd not in dp: continue
            buy=float(OA[dp[bd]]); pc=float(cA[r])
            if buy<=0 or pc<=0: continue
            fp=dpos_m[bd]; fdays=tdays[fp+1:fp+1+LOOKFWD]
            ph=np.full(LOOKFWD,np.nan,np.float32); pl=ph.copy(); pcx=ph.copy()
            for j,fd in enumerate(fdays):
                if fd in dp: k=dp[fd]; ph[j]=HA[k]/buy; pl[j]=LA[k]/buy; pcx[j]=cA[k]/buy
            if np.isnan(pcx).all(): continue
            v5=vol5A[r]; volr=float(VA[r]/v5) if vs_ok(v5) else 0.0
            recent_high=np.nanmax(cA[max(0,r-7):r]) if r>=1 else c
            pb7=(c-recent_high)/recent_high*100 if recent_high>0 else 0.0
            hh=HA[r]; llv=LA[r]; rng=hh-llv; clloc=(c-llv)/rng if rng>0 else 0.5
            F["day"].append(dpos_m[pv]); F["gap"].append((buy-pc)/pc*100)
            F["c_ma20"].append((c/m20-1)*100); F["c_ma60"].append((c/m60-1)*100)
            F["ma20_60"].append((m20/m60-1)*100); F["ma5_20"].append((m5/m20-1)*100)
            F["up120"].append(c>m120)
            F["rsi2"].append(r2A[r]); F["rsi14"].append(r14A[r])
            F["ret1"].append((c-cA[r-1])/cA[r-1]*100); F["ret5"].append((c-cA[r-5])/cA[r-5]*100); F["ret20"].append(ret20)
            F["dist_hi20"].append((c/hi20A[r]-1)*100 if hi20A[r]>0 else 0.0)
            F["dist_lo20"].append((c/lo20A[r]-1)*100 if lo20A[r]>0 else 0.0)
            F["down3"].append(int(down3A[r]) if not np.isnan(down3A[r]) else 0)
            F["volr"].append(volr); F["avgval"].append(float(avgvA[r]) if not np.isnan(avgvA[r]) else 0.0)
            F["atrp"].append(float(atrpA[r]) if not np.isnan(atrpA[r]) else 0.0)
            F["cl"].append(clloc); F["pb7"].append(pb7)
            hi.append(ph); lo.append(pl); clo.append(pcx)
    print(f"  -> 로드 {n} / 후보행 {len(F['day'])} / {time.time()-t0:.0f}s")
    nd=len(tdays); rs80=np.full(nd,np.inf)
    for d,mp in rs_by_day.items():
        v=list(mp.values())
        if v: rs80[dpos_m[d]]=np.percentile(v,80)
    out=dict(mk_strict=mk_s, mk_mild=mk_m, rs80=rs80,
             hi=np.array(hi,np.float32), lo=np.array(lo,np.float32), clo=np.array(clo,np.float32))
    types={"day":np.int32,"up120":bool,"down3":np.int8,"avgval":np.float64}
    for k,v in F.items():
        out[k]=np.array(v, types.get(k,np.float32))
    np.savez_compressed(CACHE, **out)
    print(f"[캐시 저장] {CACHE}  (행 {len(F['day'])})")

def vs_ok(x):
    try: return (not np.isnan(x)) and x>0
    except: return False

if __name__=="__main__":
    if os.path.exists(CACHE):
        print("이미 존재:", CACHE)
    else:
        build()
    print("완료 / 총 done")
