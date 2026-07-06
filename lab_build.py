# -*- coding: utf-8 -*-
"""
전략 실험실용 원시 패널: 시총필터 전 종목 5년 OHLCV를 통째로 캐시.
 남은 전략군(낙주/VCP/거래대금폭증/캘린더) 일괄 검증용. 체크포인트 재개 지원.
"""
import sys, os, time, pickle
import backtest_relax as B      # plotly 스텁 + fdr
import numpy as np, pandas as pd
from datetime import datetime, timedelta
fdr=B.fdr
END=datetime.today().strftime("%Y-%m-%d")
START=(datetime.today()-timedelta(days=365*5+300)).strftime("%Y-%m-%d")
MCMIN=B.MARKET_CAP_MIN; MCMAX=B.MARKET_CAP_MAX
PARTIAL="lab_partial.pkl"; CACHE="lab_cache.pkl"

def build():
    t0=time.time()
    ks=fdr.DataReader("KS11",START,END); ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
    tdays=sorted(ks.index.tolist())
    print(f"거래일 {len(tdays)} {tdays[0]}~{tdays[-1]}")
    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb")); tickers=st["tickers"]; name_map=st["name_map"]; data=st["data"]; start=st["done"]
        print(f"[재개] {start}/{len(tickers)}"); sys.stdout.flush()
    else:
        alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
        filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
        tickers=filt["Code"].tolist(); name_map=dict(zip(filt["Code"],filt["Name"]))
        data={}; start=0
        print(f"종목 {len(tickers)}개 수집 시작"); sys.stdout.flush()
    for ii in range(start,len(tickers)):
        tk=tickers[ii]
        if ii%150==0 and ii>start:
            pickle.dump({"tickers":tickers,"name_map":name_map,"data":data,"done":ii},open(PARTIAL,"wb"))
            print(f"  [체크포인트] {ii}/{len(tickers)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START,END)
            if df.empty or len(df)<300: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
            data[tk]={"dates":np.array(df.index),
                      "O":df["Open"].values.astype(np.float32),
                      "H":df["High"].values.astype(np.float32),
                      "L":df["Low"].values.astype(np.float32),
                      "C":df["Close"].values.astype(np.float32),
                      "V":df["Volume"].values.astype(np.float64)}
        except: continue
    pickle.dump({"tdays":tdays,"name_map":name_map,"data":data},open(CACHE,"wb"))
    if os.path.exists(PARTIAL): os.remove(PARTIAL)
    print(f"[캐시 저장] {CACHE} ({len(data)}종목, {time.time()-t0:.0f}s)")

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    else: print("캐시 존재")
    print("LAB_BUILD_DONE")
