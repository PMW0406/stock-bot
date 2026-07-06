# -*- coding: utf-8 -*-
"""
트랙2 완성: (A) 추세추종 견고성 검증 + (B) 다지수 조합/레버리지 최적화.
 신호=종가 SMA 대비, 다음날(t+1) 반영, 전환비용 0.1%. FX는 무시(지수 로컬수익 기준).
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
COST=0.001

def load(sym, st):
    d=fdr.DataReader(sym, st)["Close"].dropna()
    d.index=pd.to_datetime(d.index); return d

def perf(eq, idx):
    yrs=(idx[-1]-idx[0]).days/365.25
    cagr=(eq[-1]/eq[0])**(1/yrs)-1
    peak=np.maximum.accumulate(eq); mdd=((eq-peak)/peak).min()
    dr=np.diff(eq)/eq[:-1]; sh=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    return cagr*100, mdd*100, sh

def sma_strat(close, win, lev=1.0, rising=False):
    ma=close.rolling(win).mean(); sig=(close>ma)
    if rising: sig=sig&(ma.diff(20)>0)
    ret=close.pct_change().fillna(0).values
    pos=np.zeros(len(close)); s=sig.fillna(False).astype(float).values; pos[1:]=s[:-1]
    sw=np.abs(np.diff(np.concatenate([[0],pos])))
    eq=np.cumprod(1+pos*ret*lev - sw*COST*lev)
    return eq, pos

# ── 데이터 ──
KS=load("KS11","1995-01-01"); KQ=load("KQ11","1996-07-01")
SP=load("US500","2004-12-31"); NQ=load("IXIC","2004-12-31")

print("="*92)
print(" A-1. 파라미터 민감도 (SMA 길이 바꿔도 결과 안정적이면=견고, 과최적화 아님)")
print("="*92)
for nm,c in [("KOSPI",KS),("S&P500",SP),("NASDAQ",NQ)]:
    print(f"\n {nm}: SMA길이별 (CAGR / MDD / Sharpe)")
    row=[]
    for w in (100,120,150,180,200,220,250):
        eq,_=sma_strat(c,w); cg,md,sh=perf(eq,c.index); row.append((w,cg,md,sh))
    print("   " + " | ".join(f"SMA{w}: {cg:+4.1f}%/{md:5.1f}%/{sh:.2f}" for w,cg,md,sh in row))

print("\n"+"="*92)
print(" A-2. 시대별 견고성 (기간 반토막) — SMA200 타이밍 vs 매수후보유")
print("="*92)
for nm,c in [("KOSPI",KS),("S&P500",SP),("NASDAQ",NQ)]:
    mid=c.index[len(c)//2]
    for lo,hi,lab in [(c.index[0],mid,"전반기"),(mid,c.index[-1],"후반기")]:
        seg=c[(c.index>=lo)&(c.index<=hi)]
        eqt,_=sma_strat(seg,200); eqb,_=sma_strat(seg,1)
        ct,mt,st=perf(eqt,seg.index); cb,mb,sb=perf(eqb,seg.index)
        print(f"  {nm:7s} {lab} {str(lo.date())}~{str(hi.date())}: "
              f"타이밍 {ct:+5.1f}%/{mt:5.1f}%  vs  보유 {cb:+5.1f}%/{mb:5.1f}%")

# ── B. 조합 (공통기간 2005~, US 시작 기준) ──
df=pd.concat({"KS":KS,"SP":SP,"NQ":NQ},axis=1).sort_index().ffill().dropna()
idx=df.index
def pos_of(col,win=200,rising=False):
    c=df[col]; ma=c.rolling(win).mean(); sig=(c>ma)
    if rising: sig=sig&(ma.diff(20)>0)
    s=sig.fillna(False).astype(float).values; p=np.zeros(len(c)); p[1:]=s[:-1]; return p
rets={k:df[k].pct_change().fillna(0).values for k in df.columns}
P={k:pos_of(k,200,True) for k in df.columns}

def portfolio(weights, lev=None):
    """weights: dict asset->target비중(합<=1, 나머지 현금). 각 슬리브 독립 in/out."""
    lev=lev or {k:1.0 for k in weights}
    n=len(idx); pr=np.zeros(n); prevpos={k:0 for k in weights}
    for t in range(n):
        r=0.0
        for k,w in weights.items():
            r+=w*P[k][t]*rets[k][t]*lev[k]
            if P[k][t]!=prevpos[k]: r-=w*COST*lev[k]
            prevpos[k]=P[k][t]
        pr[t]=r
    eq=np.cumprod(1+pr); return eq

def concentrate(assets, lev=1.0):
    """켜진 자산에 균등 몰빵(없으면 현금)."""
    n=len(idx); pr=np.zeros(n); prevw={k:0 for k in assets}
    for t in range(n):
        on=[k for k in assets if P[k][t]>0]; w=1.0/len(on) if on else 0
        r=0.0
        for k in assets:
            wk=w if k in on else 0
            r+=wk*rets[k][t]*lev
            if wk!=prevw[k]: r-=abs(wk-prevw[k])*COST*lev
            prevw[k]=wk
        pr[t]=r
    return np.cumprod(1+pr)

print("\n"+"="*92)
print(f" B. 조합 포트폴리오 ({idx[0].date()}~{idx[-1].date()}, SMA200&상승 타이밍)")
print("="*92)
print(f"  {'구성':30s} | {'CAGR':>6} {'MDD':>7} {'Sharpe':>6}")
combos={
 "나스닥 단독":              portfolio({"NQ":1.0}),
 "S&P 단독":               portfolio({"SP":1.0}),
 "코스피 단독":              portfolio({"KS":1.0}),
 "정적EW(NQ/SP/KS 1/3씩)":  portfolio({"NQ":1/3,"SP":1/3,"KS":1/3}),
 "정적(NQ50/SP30/KS20)":    portfolio({"NQ":0.5,"SP":0.3,"KS":0.2}),
 "켜진자산 균등몰빵(NQ/SP/KS)": concentrate(["NQ","SP","KS"]),
 "NQ60/KS40":              portfolio({"NQ":0.6,"KS":0.4}),
}
for nm,eq in combos.items():
    cg,md,sh=perf(eq,idx); print(f"  {nm:30s} | {cg:+5.1f}% {md:6.1f}% {sh:6.2f}")
# 레버리지 변형
for label,eq in [
    ("정적EW + 나스닥슬리브2x", portfolio({"NQ":1/3,"SP":1/3,"KS":1/3},lev={"NQ":2.0,"SP":1.0,"KS":1.0})),
    ("켜진자산몰빵 전체2x",       concentrate(["NQ","SP","KS"],lev=2.0)),
]:
    cg,md,sh=perf(eq,idx); print(f"  {label:30s} | {cg:+5.1f}% {md:6.1f}% {sh:6.2f}")
# 벤치 (타이밍 없이 EW 보유)
be=np.cumprod(1+sum(rets[k] for k in df.columns)/3)
cg,md,sh=perf(be,idx); print(f"  {'[벤치] EW 매수후보유(타이밍X)':30s} | {cg:+5.1f}% {md:6.1f}% {sh:6.2f}")

# 최적 조합 연도별
print("\n [정적EW(1/3씩) SMA200&상승] 연도별 수익률")
eqEW=combos["정적EW(NQ/SP/KS 1/3씩)"]
ser=pd.Series(eqEW,index=idx); yr=ser.groupby(ser.index.year).apply(lambda x:x.iloc[-1]/x.iloc[0]-1)*100
print("   "+"  ".join(f"{y}:{v:+.0f}%" for y,v in yr.items()))
print("="*92); print("완료 / 총 done")
