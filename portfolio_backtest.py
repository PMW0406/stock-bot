# -*- coding: utf-8 -*-
"""
포트폴리오 복리 시뮬레이션 — '수익 최대' 설정 탐색.
 실제 Q2 신호(backtest_hold_cache60.npz: 진짜 게이트 + 60일 저가/종가 경로) 사용.
 청산: 승자 태우기(조기 익절 없음) + 손절 S + 시간 H.
 자금운용: 최대 K종목 동시보유, 매수시 equity/K 균등, 왕복수수료 COST 반영, 복리.
 출력: 총수익률(≈1년), MDD, 거래수, 실현승률.
"""
import numpy as np, itertools
Z=np.load("backtest_hold_cache60.npz")
mkm=Z["mk_mild"]; rs80=Z["rs80"]
day=Z["day"]; gap=Z["gap"]; c20=Z["c20"]; c2060=Z["c2060"]; pb=Z["pb"]; vr=Z["vr"]
cl=Z["cl"]; be=Z["be"]; av=Z["av"]; rs=Z["rs"]; sc=Z["sc"]; wok=Z["wok"]; lo=Z["lo"]; clo=Z["clo"]
MTV=3_000_000_000; COST=0.003   # 왕복 0.3% (수수료+세금+슬리피지 근사)
score=sc+np.where(wok,5,0)
ndays=len(mkm)

# 진짜 Q2 신호 마스크
Q2 = mkm[day]&c20&c2060&(pb>=-8)&(pb<=-0.5)&(vr>=0.8)&(vr<=3.0)&~be&(cl>=0.40)&(cl<=0.85)&(av>=MTV)&wok&(score>=70)&(rs>=rs80[day])&(gap<2.0)
idxs=np.where(Q2)[0]
print(f"Q2 신호 {len(idxs)}건\n")

def trade_outcomes(stop, hold):
    """각 신호의 (entry_idx, exit_idx, ret_net, score) 리스트. 승자태우기(무목표)."""
    out=[]
    for i in idxs:
        L=lo[i,:hold]; C=clo[i,:hold]
        v=~np.isnan(C)
        if not v.any(): continue
        exit_off=None; ret=None
        if stop is not None:
            sp=1-stop
            hit=np.where(v & (L<=sp))[0]
            if len(hit): exit_off=int(hit[0]); ret=-stop
        if exit_off is None:
            last=np.where(v)[0][-1]; exit_off=int(last); ret=float(C[last]-1)
        ret-=COST
        d=int(day[i]); out.append((d+1, d+2+exit_off, ret, float(score[i])))
    return out

def portfolio(trades, K):
    by_entry={}
    for e,x,r,s in trades: by_entry.setdefault(e,[]).append((s,x,r))
    cash=1.0; positions=[]  # (exit_idx,size,ret)
    eq_series=[]
    for t in range(ndays+62):
        # 청산
        still=[]
        for x,size,r in positions:
            if x==t: cash+=size*(1+r)
            else: still.append((x,size,r))
        positions=still
        eq=cash+sum(sz for _,sz,_ in positions)
        eq_series.append(eq)
        # 신규 진입 (점수 높은 순)
        cand=sorted(by_entry.get(t,[]), key=lambda z:z[0], reverse=True)
        for s,x,r in cand:
            if len(positions)>=K: break
            size=eq/K
            if 0<size<=cash+1e-12:
                cash-=size; positions.append((x,size,r))
        # eq 재계산 불필요(다음 루프서)
    final=cash+sum(sz for _,sz,_ in positions)
    eqs=np.array(eq_series); peak=np.maximum.accumulate(eqs); mdd=((eqs-peak)/peak).min()*100
    return final-1, mdd

def wr_of(trades):
    r=np.array([t[2] for t in trades]);
    return (r>0).mean()*100, r.mean()*100, len(r)

print(f"{'손절':>6} {'보유':>4} {'K':>3} | {'총수익률':>8} {'MDD':>7} {'거래수':>5} {'실현승률':>7} {'평균/거래':>8}")
for stop in (None, 0.10, 0.15, 0.20):
    for hold in (10, 20, 40):
        tr=trade_outcomes(stop,hold)
        wr,avg,n=wr_of(tr)
        for K in (5, 8):
            tot,mdd=portfolio(tr,K)
            stag="무손절" if stop is None else f"-{int(stop*100)}%"
            print(f"{stag:>6} {hold:>3}일 {K:>3} | {tot*100:+7.1f}% {mdd:6.1f}% {n:5d} {wr:6.1f}% {avg:+7.2f}%")
    print()
print("완료 / 총 done")
