# -*- coding: utf-8 -*-
"""
research_cache.npz 로 서로 다른 전략을 실현손익 기준 비교.
목표: 기존 추세눌림목에 매몰되지 않고, '승률이 엣지로 나오는' 진입을 찾기.
공통: 유동성 거래대금30억↑, 진입=신호 다음날 시가, 같은날 목표·손절 동시=손절우선(보수).
"""
import sys, os
import numpy as np
CACHE="research_cache.npz"
if not os.path.exists(CACHE): print("캐시 없음 — research_build.py 먼저"); sys.exit()
Z=np.load(CACHE)
day=Z["day"]; gap=Z["gap"]; c_ma20=Z["c_ma20"]; c_ma60=Z["c_ma60"]; ma20_60=Z["ma20_60"]; ma5_20=Z["ma5_20"]
up120=Z["up120"]; rsi2=Z["rsi2"]; rsi14=Z["rsi14"]; ret1=Z["ret1"]; ret5=Z["ret5"]; ret20=Z["ret20"]
dist_hi20=Z["dist_hi20"]; dist_lo20=Z["dist_lo20"]; down3=Z["down3"]; volr=Z["volr"]; avgval=Z["avgval"]
atrp=Z["atrp"]; cl=Z["cl"]; pb7=Z["pb7"]
mk_s=Z["mk_strict"]; mk_m=Z["mk_mild"]; rs80=Z["rs80"]; hi=Z["hi"]; lo=Z["lo"]; clo=Z["clo"]
MTV=3_000_000_000
liq = avgval>=MTV
print(f"총 후보행 {len(day)} / 유동성통과 {int(liq.sum())}\n")

def sim(mask, target, stop, hold):
    idx=np.where(mask)[0]
    if len(idx)==0: return None
    H=hi[idx][:,:hold]; L=lo[idx][:,:hold]; C=clo[idx][:,:hold]
    n=len(idx); ret=np.full(n,np.nan); done=np.zeros(n,bool); tg=1+target; sp=1-stop
    for j in range(hold):
        hj=H[:,j]; lj=L[:,j]; cj=C[:,j]; v=~np.isnan(cj)
        s=v&~done&(lj<=sp); ret[s]=-stop; done[s]=True
        if target<5:
            t=v&~done&(hj>=tg); ret[t]=target; done[t]=True
    for i in np.where(~done)[0]:
        cv=C[i][~np.isnan(C[i])]
        if len(cv): ret[i]=cv[-1]-1; done[i]=True
    return ret[done]*100

def rank(mask):
    """RS 상위20% (그날 기준) 필터 적용된 mask 반환"""
    return mask & (ret20>=rs80[day])

def rep(tag, mask, policies):
    n=int(mask.sum())
    if n==0: print(f"  {tag}: 0건"); return
    wk=n/52.0
    print(f"  {tag}  (신호 {n}건, 주{wk:.1f})")
    for pname,tg,sp,hd in policies:
        r=sim(mask,tg,sp,hd)
        if r is None or len(r)==0: print(f"      {pname}: 0"); continue
        win=r>0; wr=win.mean()*100; avg=r.mean(); med=np.median(r)
        aw=r[win].mean() if win.any() else 0; al=r[~win].mean() if (~win).any() else 0
        pf=abs(win.sum()*aw/((~win).sum()*al)) if (~win).any() and al!=0 else 99
        print(f"      {pname:22s} 승률{wr:5.1f}% 평균{avg:+6.2f}% 중앙{med:+6.2f}% PF{pf:5.2f}")

# ── 전략 정의 ─────────────────────────────
# 역추세/과매도 반등류는 '작은목표+짧은보유'가 정석
MR = [("+3%/-6%/5d",0.03,0.06,5),("+4%/-8%/5d",0.04,0.08,5),
      ("+5%/-8%/7d",0.05,0.08,7),("목표없음/-8%/5d",9.9,0.08,5)]
TR = [("목표없음/-10%/10d",9.9,0.10,10),("목표없음/-15%/10d",9.9,0.15,10),
      ("+10%/-8%/10d",0.10,0.08,10)]

print("="*96)
print(" A. [기준] 추세 눌림목 (Q2류): 상승추세 + 얕은눌림 + RS상위20%")
A = liq & (c_ma20>0) & (ma20_60>0) & (pb7>=-8)&(pb7<=-0.5) & (cl>=0.40)&(cl<=0.85) & (mk_m[day])
rep("A", rank(A), TR)

print("\n B. [역추세] RSI2 과매도 반등 (Connors류): 장기상승(120MA↑) + RSI2<10")
B1 = liq & up120 & (c_ma60>0) & (rsi2<10)
rep("B1 RSI2<10", B1, MR)
B2 = liq & up120 & (c_ma60>0) & (rsi2<5)
rep("B2 RSI2<5 (더 과매도)", B2, MR)
B3 = liq & up120 & (c_ma60>0) & (rsi2<10) & (c_ma20<0)   # MA20 아래로 눌린 것만
rep("B3 RSI2<10 & MA20아래", B3, MR)

print("\n C. [역추세] 연속하락 후 반등: 상승추세 + 최근3일중 하락2+ + 단기급락")
C1 = liq & up120 & (c_ma60>0) & (down3>=2) & (ret5<=-4)
rep("C1 down3>=2 & ret5<=-4%", C1, MR)

print("\n D. [과매도 광의] RSI14<30 + 장기상승추세")
D1 = liq & up120 & (rsi14<30)
rep("D1 RSI14<30 & 120MA↑", D1, MR)
D2 = liq & up120 & (rsi14<25) & (c_ma60>0)
rep("D2 RSI14<25 & 60MA↑", D2, MR)

print("\n E. [혼합] RSI2<10 + RS상위20% (질 필터 추가)")
rep("E RSI2<10 & RS상위20%", rank(B1), MR)
print("="*96)
print("완료 / 총 done")
