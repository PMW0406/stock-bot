# -*- coding: utf-8 -*-
"""
팩터 스캔용 5년 캐시: 강세장(코스피120MA위) 전 종목/전 거래일에 다수 팩터 + 10일 forward수익.
 매수=신호 다음날 시가, 매도=10거래일 후 종가. 체크포인트로 재개.
"""
import sys, os, time, pickle
import backtest_relax as B
import numpy as np, pandas as pd
from datetime import datetime, timedelta
fdr=B.fdr; MTV=B.MIN_TRADING_VALUE
END=datetime.today().strftime("%Y-%m-%d")
START_MKT=(datetime.today()-timedelta(days=365*5)).strftime("%Y-%m-%d")
START_STK=(datetime.today()-timedelta(days=365*5+320)).strftime("%Y-%m-%d")
MCMIN=B.MARKET_CAP_MIN; MCMAX=B.MARKET_CAP_MAX
HOLD=10; PARTIAL="factor_partial.pkl"; CACHE="factor_cache.npz"

def rsi(close,p):
    d=close.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ru=up.ewm(alpha=1/p,adjust=False).mean(); rd=dn.ewm(alpha=1/p,adjust=False).mean()
    return (100-100/(1+ru/rd.replace(0,np.nan))).fillna(50)

FEATS=["ret5","ret20","ret60","rsi2","rsi14","dist_hi20","dist_lo20","dist_hi52",
       "atrp","volr","c_ma20","c_ma60","ma20_60","ma5_20","down3","avgval"]

def build():
    t0=time.time(); print("지수 5년..."); sys.stdout.flush()
    ks=fdr.DataReader("KS11",START_MKT,END); ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
    tdays=sorted(ks.index.tolist()); dpm={d:i for i,d in enumerate(tdays)}
    ksc=ks["Close"].reindex(tdays).ffill(); reg=(ksc>ksc.rolling(120).mean()).fillna(False).values
    print(f"거래일 {len(tdays)} / 강세장 {int(reg.sum())}일")
    prev_of={bd:(tdays[i-1] if i>0 else None) for i,bd in enumerate(tdays)}
    valid_buy=tdays[:-(HOLD+1)]
    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb")); tickers=st["tickers"]; col=st["col"]; start=st["done"]
        print(f"[재개] {start}/{len(tickers)} (행{len(col['day'])})"); sys.stdout.flush()
    else:
        alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
        filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
        tickers=filt["Code"].tolist()
        col={k:[] for k in (["day","ret10"]+FEATS)}; start=0
        print(f"종목 {len(tickers)}개 스캔..."); sys.stdout.flush()
    for ii in range(start,len(tickers)):
        tk=tickers[ii]
        if ii%150==0 and ii>start:
            pickle.dump({"tickers":tickers,"col":col,"done":ii},open(PARTIAL,"wb"))
            print(f"  [체크포인트] {ii}/{len(tickers)} ({time.time()-t0:.0f}s, 행{len(col['day'])})"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_STK,END)
            if df.empty or len(df)<260: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        C=df["Close"]; Hh=df["High"]; Ll=df["Low"]; Vv=df["Volume"]; Ov=df["Open"]
        ma5=C.rolling(5).mean(); ma20=C.rolling(20).mean(); ma60=C.rolling(60).mean()
        r2=rsi(C,2); r14=rsi(C,14)
        pc=C.shift(1); tr=np.maximum(Hh-Ll,np.maximum((Hh-pc).abs(),(Ll-pc).abs()))
        atr=tr.ewm(alpha=1/14,adjust=False).mean()
        hi20=Hh.rolling(20).max(); lo20=Ll.rolling(20).min(); hi52=Hh.rolling(252).max()
        avgval=(Vv*C).rolling(20).mean(); vol5=Vv.rolling(5).mean()
        down3=(C.diff()<0).astype(int).rolling(3).sum()
        cA=C.values; ma5A=ma5.values; ma20A=ma20.values; ma60A=ma60.values
        r2A=r2.values; r14A=r14.values; atrA=atr.values; hi20A=hi20.values; lo20A=lo20.values
        hi52A=hi52.values; avgvA=avgval.values; vol5A=vol5.values; down3A=down3.values
        VA=Vv.values; OA=Ov.values
        dates=list(df.index); dp={d:j for j,d in enumerate(dates)}
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dp: continue
            r=dp[pv]
            if r<252: continue
            di=dpm[pv]
            if not reg[di]: continue
            if bd not in dp: continue
            fp=dpm[bd]
            if fp+HOLD>=len(tdays): continue
            sd=tdays[fp+HOLD]
            if sd not in dp: continue
            buy=float(OA[dp[bd]]); sell=float(cA[dp[sd]])
            if buy<=0: continue
            m20=ma20A[r]; m60=ma60A[r]; m5=ma5A[r]; c=cA[r]
            if m20<=0 or m60<=0 or m5<=0 or np.isnan(m60): continue
            v5=vol5A[r]; av=avgvA[r]
            if np.isnan(av) or av<MTV: continue      # 유동성 공통
            col["day"].append(di); col["ret10"].append((sell-buy)/buy*100)
            col["ret5"].append((c-cA[r-5])/cA[r-5]*100)
            col["ret20"].append((c-cA[r-20])/cA[r-20]*100)
            col["ret60"].append((c-cA[r-60])/cA[r-60]*100)
            col["rsi2"].append(r2A[r]); col["rsi14"].append(r14A[r])
            col["dist_hi20"].append((c/hi20A[r]-1)*100 if hi20A[r]>0 else 0)
            col["dist_lo20"].append((c/lo20A[r]-1)*100 if lo20A[r]>0 else 0)
            col["dist_hi52"].append((c/hi52A[r]-1)*100 if hi52A[r]>0 else 0)
            col["atrp"].append(atrA[r]/c*100 if c>0 else 0)
            col["volr"].append(float(VA[r]/v5) if v5>0 else 0)
            col["c_ma20"].append((c/m20-1)*100); col["c_ma60"].append((c/m60-1)*100)
            col["ma20_60"].append((m20/m60-1)*100); col["ma5_20"].append((m5/m20-1)*100)
            col["down3"].append(int(down3A[r]) if not np.isnan(down3A[r]) else 0)
            col["avgval"].append(float(av))
    print(f"  -> 행 {len(col['day'])} / {time.time()-t0:.0f}s")
    out={"tdays":np.array(tdays),"day":np.array(col["day"],np.int32),"ret10":np.array(col["ret10"],np.float32)}
    for k in FEATS: out[k]=np.array(col[k],np.float32)
    np.savez_compressed(CACHE, **out)
    if os.path.exists(PARTIAL): os.remove(PARTIAL)
    print(f"[캐시 저장] {CACHE}")

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    print("완료 / 총 done")
