# -*- coding: utf-8 -*-
"""
트랙2 심층: 지수 추세추종 (20~31년, 다지수 × 이평 × 상승필터 × 레버리지).
 실행: 종가로 신호 → 다음날(t+1) 반영(룩어헤드 방지). 포지션 0/1.
 비용: 전환 1회당 SWITCH_COST. 지표: CAGR·MDD·Sharpe·시장체류%·전환수·연도별.
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
import numpy as np, pandas as pd, FinanceDataReader as fdr
SWITCH_COST=0.001   # 전환 1회 0.1%

def metrics(eq, idx):
    eq=np.asarray(eq); n=len(eq)
    yrs=(idx[-1]-idx[0]).days/365.25
    cagr=(eq[-1]/eq[0])**(1/yrs)-1
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()
    dr=np.diff(eq)/eq[:-1]
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    return cagr*100, mdd*100, sharpe

def run(close, sig, lev=1.0):
    """sig: bool array (해당 종가 기준 '보유' 조건). 다음날 반영."""
    ret=close.pct_change().fillna(0).values
    pos=np.zeros(len(close))
    s=sig.astype(float).values
    pos[1:]=s[:-1]   # t+1 반영
    strat=pos*ret*lev
    # 전환비용
    switch=np.abs(np.diff(np.concatenate([[0],pos])))
    cost=switch*SWITCH_COST*lev
    eq=np.cumprod(1+strat-cost)
    nsw=int((np.abs(np.diff(pos))>0).sum())
    inmkt=(pos>0).mean()*100
    return eq, nsw, inmkt

def yearly(eq, idx):
    df=pd.Series(eq, index=idx)
    yr=df.groupby(df.index.year).apply(lambda x: x.iloc[-1]/x.iloc[0]-1)*100
    return yr

INDICES=[("KOSPI","KS11","1995-01-01"),("KOSDAQ","KQ11","1996-07-01"),
         ("S&P500","US500","2004-12-31"),("NASDAQ","IXIC","2004-12-31")]

for nm,sym,st in INDICES:
    d=fdr.DataReader(sym,st); c=d["Close"].dropna()
    idx=c.index
    ma100=c.rolling(100).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    rise200=ma200.diff(20)>0
    strat={
      "매수후보유(B&H)": pd.Series(True,index=c.index),
      "SMA100 위":  c>ma100,
      "SMA150 위":  c>ma150,
      "SMA200 위":  c>ma200,
      "SMA200 위&상승": (c>ma200)&rise200,
    }
    print("\n"+"="*100)
    print(f" {nm} ({sym})  {idx[0].date()}~{idx[-1].date()}  [{(idx[-1]-idx[0]).days/365.25:.0f}년]")
    print("="*100)
    print(f"  {'전략':16s} | {'CAGR':>6} {'MDD':>7} {'Sharpe':>6} {'시장체류':>7} {'전환수':>5}")
    results={}
    for lname,sig in strat.items():
        eq,nsw,inm=run(c,sig.fillna(False))
        cagr,mdd,sh=metrics(eq,idx)
        results[lname]=(eq,cagr,mdd,sh)
        print(f"  {lname:16s} | {cagr:+5.1f}% {mdd:6.1f}% {sh:6.2f} {inm:6.1f}% {nsw:5d}")
    # 2배 레버리지 (SMA200 위)
    eq2,nsw2,inm2=run(c,(c>ma200).fillna(False),lev=2.0)
    c2,m2,s2=metrics(eq2,idx)
    print(f"  {'SMA200위 x2레버':16s} | {c2:+5.1f}% {m2:6.1f}% {s2:6.2f} {inm2:6.1f}% {nsw2:5d}")

print("\n"+"="*100)
print(" [KOSPI 연도별 수익률] B&H vs SMA200 타이밍  (약세장 방어 확인)")
print("="*100)
d=fdr.DataReader("KS11","1998-01-01"); c=d["Close"].dropna(); idx=c.index
ma200=c.rolling(200).mean()
eqb,_,_=run(c,pd.Series(True,index=c.index))
eqt,_,_=run(c,(c>ma200).fillna(False))
yb=yearly(eqb,idx); yt=yearly(eqt,idx)
yrs=sorted(set(yb.index))
print("  연도  B&H     SMA200타이밍")
for y in yrs:
    print(f"  {y}  {yb.get(y,0):+6.1f}%  {yt.get(y,0):+6.1f}%")
print("="*100)
print("완료 / 총 done")
