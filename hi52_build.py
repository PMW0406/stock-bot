# -*- coding: utf-8 -*-
"""
52주 신고가 전략용 5년 캐시: 강세장 + 유동성 + 신고가 -10% 이내 후보의
 15일 가격경로(고/저/종) 저장 → 손절·보유·청산 정책 시뮬 가능.
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
LOOKFWD=15; NEAR=-10.0
PARTIAL="hi52_partial.pkl"; CACHE="hi52_cache.npz"

def build():
    t0=time.time(); print("지수 5년..."); sys.stdout.flush()
    ks=fdr.DataReader("KS11",START_MKT,END); ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
    tdays=sorted(ks.index.tolist()); dpm={d:i for i,d in enumerate(tdays)}
    ksc=ks["Close"].reindex(tdays).ffill(); reg=(ksc>ksc.rolling(120).mean()).fillna(False).values
    print(f"거래일 {len(tdays)} / 강세장 {int(reg.sum())}일")
    prev_of={bd:(tdays[i-1] if i>0 else None) for i,bd in enumerate(tdays)}
    valid_buy=tdays[:-2]
    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb")); tickers=st["tickers"]; col=st["col"]; hi=st["hi"]; lo=st["lo"]; clo=st["clo"]; start=st["done"]
        print(f"[재개] {start}/{len(tickers)} (행{len(col['day'])})"); sys.stdout.flush()
    else:
        alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
        filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
        tickers=filt["Code"].tolist()
        col=dict(day=[],hi52=[],ret20=[],gap=[],avgval=[]); hi=[]; lo=[]; clo=[]; start=0
        print(f"종목 {len(tickers)}개 스캔..."); sys.stdout.flush()
    for ii in range(start,len(tickers)):
        tk=tickers[ii]
        if ii%150==0 and ii>start:
            pickle.dump({"tickers":tickers,"col":col,"hi":hi,"lo":lo,"clo":clo,"done":ii},open(PARTIAL,"wb"))
            print(f"  [체크포인트] {ii}/{len(tickers)} ({time.time()-t0:.0f}s, 행{len(col['day'])})"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_STK,END)
            if df.empty or len(df)<260: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        C=df["Close"]; Hh=df["High"]; Ll=df["Low"]; Vv=df["Volume"]; Ov=df["Open"]
        hi52s=Hh.rolling(252).max(); avgval=(Vv*C).rolling(20).mean()
        cA=C.values; HA=Hh.values; LA=Ll.values; OA=Ov.values; h52A=hi52s.values; avA=avgval.values
        dates=list(df.index); dp={d:j for j,d in enumerate(dates)}
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dp: continue
            r=dp[pv]
            if r<252: continue
            di=dpm[pv]
            if not reg[di]: continue
            c=cA[r]; h52=h52A[r]; av=avA[r]
            if np.isnan(h52) or h52<=0 or np.isnan(av) or av<MTV: continue
            d52=(c/h52-1)*100
            if d52<NEAR: continue                     # 신고가 -10% 이내만
            if bd not in dp: continue
            buy=float(OA[dp[bd]]); pc=float(cA[r])
            if buy<=0 or pc<=0: continue
            gap=(buy-pc)/pc*100
            fp=dpm[bd]; fdays=tdays[fp+1:fp+1+LOOKFWD]
            ph=np.full(LOOKFWD,np.nan,np.float32); pl=ph.copy(); pcx=ph.copy()
            for j,fd in enumerate(fdays):
                if fd in dp: k=dp[fd]; ph[j]=HA[k]/buy; pl[j]=LA[k]/buy; pcx[j]=cA[k]/buy
            if np.isnan(pcx).all(): continue
            ret20=(c-cA[r-20])/cA[r-20]*100
            col["day"].append(di); col["hi52"].append(d52); col["ret20"].append(ret20)
            col["gap"].append(gap); col["avgval"].append(float(av))
            hi.append(ph); lo.append(pl); clo.append(pcx)
    print(f"  -> 행 {len(col['day'])} / {time.time()-t0:.0f}s")
    np.savez_compressed(CACHE, tdays=np.array(tdays),
        day=np.array(col["day"],np.int32), hi52=np.array(col["hi52"],np.float32),
        ret20=np.array(col["ret20"],np.float32), gap=np.array(col["gap"],np.float32),
        avgval=np.array(col["avgval"],np.float64),
        hi=np.array(hi,np.float32), lo=np.array(lo,np.float32), clo=np.array(clo,np.float32))
    if os.path.exists(PARTIAL): os.remove(PARTIAL)
    print(f"[캐시 저장] {CACHE}")

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    print("완료 / 총 done")
