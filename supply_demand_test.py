# -*- coding: utf-8 -*-
"""
트랙1 탐색: 수급(외국인+기관 순매수) → 이후 수익 관계.
 ⚠️ investor_data.parquet가 2026-03~06(3개월, 대세강세장)뿐 → 결과는 개념검증용,
    국면검증 불가(강세장선 뭘 사도 오름). 표본 400종목.
"""
import sys,types
class _P:
    def __getattr__(s,k): return _p
    def __setattr__(s,k,v): pass
    def __call__(s,*a,**k): return _p
_p=_P()
pl=types.ModuleType("plotly"); pl.__path__=[]; pl.__getattr__=lambda k:_p; sys.modules["plotly"]=pl
for s in ("io","graph_objects","express","subplots","offline","figure_factory"):
    m=types.ModuleType("plotly."+s); m.__getattr__=lambda k:_p; setattr(pl,s,m); sys.modules["plotly."+s]=m
import numpy as np, pandas as pd, FinanceDataReader as fdr, time

iv=pd.read_parquet("investor_data.parquet")
iv["date"]=pd.to_datetime(iv["date"]).dt.strftime("%Y-%m-%d")
tickers=sorted(iv["ticker"].unique())
tickers=tickers[::max(1,len(tickers)//400)]   # ~400 표본
print(f"수급기간 {iv['date'].min()}~{iv['date'].max()} / 표본 {len(tickers)}종목")
HOLD=5
prices={}
t0=time.time()
for i,tk in enumerate(tickers):
    if i%100==0: print(f"  price {i}/{len(tickers)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
    try:
        d=fdr.DataReader(tk,"2026-03-01","2026-07-02")
        if len(d)>HOLD+2: prices[tk]=d
    except: pass

rows=[]
for tk,d in prices.items():
    d=d.copy(); d.index=pd.to_datetime(d.index).strftime("%Y-%m-%d")
    dates=list(d.index); O=d["Open"].values; C=d["Close"].values; pos={x:j for j,x in enumerate(dates)}
    sub=iv[iv["ticker"]==tk]
    for _,r in sub.iterrows():
        dt=r["date"]
        if dt not in pos: continue
        j=pos[dt]
        if j+1+HOLD>=len(dates): continue
        buy=O[j+1]                       # 신호 다음날 시가
        fut=C[j+1+HOLD]                  # HOLD일 후 종가
        if buy<=0: continue
        rows.append((tk, dt, r["frgn"], r["inst"], (fut-buy)/buy*100))
df=pd.DataFrame(rows, columns=["tk","date","frgn","inst","fwd5"])
print(f"\n관측 {len(df)}건")
df["flow"]=df["frgn"]+df["inst"]

# 1) 수급 5분위별 이후 5일 수익
df["q"]=pd.qcut(df["flow"].rank(method="first"),5,labels=["Q1(매도최다)","Q2","Q3","Q4","Q5(매수최다)"])
print("\n[순매수 5분위별 이후 5일 평균수익]")
g=df.groupby("q")["fwd5"]
for k in g.groups: print(f"  {k:12s}: 평균 {g.get_group(k).mean():+5.2f}%  승률 {(g.get_group(k)>0).mean()*100:4.1f}%  n{len(g.get_group(k))}")
print(f"  전체평균: {df['fwd5'].mean():+.2f}% (강세장 기저효과)")

# 2) 외국인·기관 동시 순매수 vs 동시 순매도
both_buy=df[(df["frgn"]>0)&(df["inst"]>0)]
both_sell=df[(df["frgn"]<0)&(df["inst"]<0)]
print("\n[동시 순매수 vs 동시 순매도]")
print(f"  둘다매수: 평균 {both_buy['fwd5'].mean():+.2f}% 승률 {(both_buy['fwd5']>0).mean()*100:.1f}% n{len(both_buy)}")
print(f"  둘다매도: 평균 {both_sell['fwd5'].mean():+.2f}% 승률 {(both_sell['fwd5']>0).mean()*100:.1f}% n{len(both_sell)}")
print(f"  스프레드(매수-매도): {both_buy['fwd5'].mean()-both_sell['fwd5'].mean():+.2f}%p  <- 이게 +면 수급 엣지 존재")
print("완료 / 총 done")
