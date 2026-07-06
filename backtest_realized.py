# -*- coding: utf-8 -*-
"""
실현손익 백테스트 — 실제 매매규칙 반영
 진입: 신호 다음날 시가
 청산: (a) 보유중 고가가 +목표 → 목표가 익절
       (b) 보유중 저가가 -손절(4%) → 손절 (같은날 둘다면 보수적으로 손절 우선)
       (c) 미도달 시 HOLD_DAYS째 종가 청산
 V0(현행 strict+엄격볼륨) vs Q2(mild+완화볼륨) 을 여러 목표가 정책으로 비교.
 * 수수료/세금/슬리피지 미반영. 보유기간 가격경로를 캐시(.npz)에 저장 → 재실행 즉시.
"""
import sys, os, time
import backtest_relax as B          # plotly 스텁 + 헬퍼(compute_metrics, build_weekly_cache, market_states) 재사용
import numpy as np, pandas as pd

fdr = B.fdr
HOLD_DAYS        = B.HOLD_DAYS
MIN_TRADING_VALUE= B.MIN_TRADING_VALUE
START_DATE       = B.START_DATE
START_DATE_STOCK = B.START_DATE_STOCK
END_DATE         = B.END_DATE
MARKET_CAP_MIN   = B.MARKET_CAP_MIN
MARKET_CAP_MAX   = B.MARKET_CAP_MAX
CACHE = "backtest_realized_cache.npz"

def build():
    t0=time.time()
    print("시장 지수..."); sys.stdout.flush()
    kospi=fdr.DataReader("KS11",START_DATE,END_DATE); kosdaq=fdr.DataReader("KQ11",START_DATE,END_DATE)
    kospi.index=pd.to_datetime(kospi.index).strftime("%Y-%m-%d"); kosdaq.index=pd.to_datetime(kosdaq.index).strftime("%Y-%m-%d")
    trading_days=sorted(kospi.index.tolist()); day_pos={d:i for i,d in enumerate(trading_days)}
    mk_strict,mk_mild=B.market_states(trading_days,kospi,kosdaq)
    print(f"시장 strict={int(mk_strict.sum())} mild={int(mk_mild.sum())}/{len(trading_days)}")

    alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
    filt=alls[(alls["Marcap"]>=MARKET_CAP_MIN)&(alls["Marcap"]<=MARKET_CAP_MAX)]
    tickers=filt["Code"].tolist()
    print(f"종목 {len(tickers)}개 로드+경로수집..."); sys.stdout.flush()

    prev_of={bd:(trading_days[i-1] if i>0 else None) for i,bd in enumerate(trading_days)}
    valid_buy=trading_days[:-HOLD_DAYS]
    C=[[] for _ in range(11)]  # day,gap, c20,c2060,pb,vr,vd,cl,be,av,rs,sc,wok -> use dict lists below
    col=dict(day=[],gap=[],c20=[],c2060=[],pb=[],vr=[],vd=[],cl=[],be=[],av=[],rs=[],sc=[],wok=[])
    hi=[]; lo=[]; clo=[]   # 보유일별 비율(High/Low/Close ÷ buy) shape (H,)
    rs_by_day={}; n=0
    for i,tk in enumerate(tickers):
        if i%300==0: print(f"  {i}/{len(tickers)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_DATE_STOCK,END_DATE)
            if df.empty or len(df)<65: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        n+=1; dpos={d:j for j,d in enumerate(df.index)}
        wc=B.build_weekly_cache(df)
        H=df["High"].values; L=df["Low"].values; CL=df["Close"].values; O=df["Open"].values
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dpos: continue
            m=B.compute_metrics(df,dpos[pv])
            if m is None: continue
            rs_by_day.setdefault(pv,{})[tk]=m[8]
            if bd not in dpos: continue
            bpos=dpos[bd]; buy=float(O[bpos]); pc=float(CL[dpos[pv]])
            if buy<=0 or pc<=0: continue
            gap=(buy-pc)/pc*100
            fp=day_pos[bd]; fdays=trading_days[fp+1:fp+1+HOLD_DAYS]
            ph=np.full(HOLD_DAYS,np.nan,np.float32); pl=ph.copy(); pcx=ph.copy()
            for j,fd in enumerate(fdays):
                if fd in dpos:
                    k=dpos[fd]; ph[j]=H[k]/buy; pl[j]=L[k]/buy; pcx[j]=CL[k]/buy
            if np.isnan(pcx).all(): continue
            c20,c2060,pb,vr,vd,cl,be,av,rs,sc=m
            col["day"].append(day_pos[pv]); col["gap"].append(gap)
            col["c20"].append(c20); col["c2060"].append(c2060); col["pb"].append(pb); col["vr"].append(vr)
            col["vd"].append(vd); col["cl"].append(cl); col["be"].append(be); col["av"].append(av)
            col["rs"].append(rs); col["sc"].append(sc); col["wok"].append(bool(wc.get(pv,False)))
            hi.append(ph); lo.append(pl); clo.append(pcx)
    print(f"  -> 로드 {n} / 후보행 {len(col['day'])} / {time.time()-t0:.0f}s")
    nd=len(trading_days); rs80=np.full(nd,np.inf)
    for d,mp in rs_by_day.items():
        v=list(mp.values())
        if v: rs80[day_pos[d]]=np.percentile(v,80)
    np.savez_compressed(CACHE,
        mk_strict=mk_strict, mk_mild=mk_mild, rs80=rs80,
        day=np.array(col["day"],np.int32), gap=np.array(col["gap"],np.float32),
        c20=np.array(col["c20"],bool), c2060=np.array(col["c2060"],bool),
        pb=np.array(col["pb"],np.float32), vr=np.array(col["vr"],np.float32), vd=np.array(col["vd"],bool),
        cl=np.array(col["cl"],np.float32), be=np.array(col["be"],bool), av=np.array(col["av"],np.float64),
        rs=np.array(col["rs"],np.float32), sc=np.array(col["sc"],np.int16), wok=np.array(col["wok"],bool),
        hi=np.array(hi,np.float32), lo=np.array(lo,np.float32), clo=np.array(clo,np.float32))
    print(f"[캐시 저장] {CACHE}")

def simulate(hi,lo,clo, target, stop):
    """벡터화 실현수익. 같은날 목표·손절 동시면 손절 우선(보수)."""
    N=hi.shape[0]; H=hi.shape[1]
    ret=np.full(N,np.nan); done=np.zeros(N,bool)
    tgt=1.0+target; stp=1.0-stop
    for j in range(H):
        hj=hi[:,j]; lj=lo[:,j]; cj=clo[:,j]
        valid=~np.isnan(cj)
        hit_stop = valid & ~done & (lj<=stp)
        ret[hit_stop]=-stop; done[hit_stop]=True
        hit_tgt = valid & ~done & (hj>=tgt)
        ret[hit_tgt]=target; done[hit_tgt]=True
    # 미청산 → 마지막 유효 종가
    for i in np.where(~done)[0]:
        row=clo[i]; v=row[~np.isnan(row)]
        if len(v): ret[i]=v[-1]-1.0; done[i]=True
    return ret[done]

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    Z=np.load(CACHE)
    mk={"strict":Z["mk_strict"],"mild":Z["mk_mild"]}; rs80=Z["rs80"]
    day=Z["day"]; gap=Z["gap"]; c20=Z["c20"]; c2060=Z["c2060"]; pb=Z["pb"]; vr=Z["vr"]; vd=Z["vd"]
    cl=Z["cl"]; be=Z["be"]; av=Z["av"]; rs=Z["rs"]; sc=Z["sc"]; wok=Z["wok"]
    hi=Z["hi"]; lo=Z["lo"]; clo=Z["clo"]
    score=sc+np.where(wok,5,0)
    print(f"후보행 {len(day)} | strict={int(mk['strict'].sum())} mild={int(mk['mild'].sum())}")

    def mask_of(variant):
        if variant=="V0":
            m=mk["strict"][day]; base=m&c20&c2060&(pb>=-8)&(pb<=-0.5)&(vr>=1.0)&(vr<=2.5)&vd
        else:  # Q2
            m=mk["mild"][day];   base=m&c20&c2060&(pb>=-8)&(pb<=-0.5)&(vr>=0.8)&(vr<=3.0)
        base=base&~be&(cl>=0.40)&(cl<=0.85)&(av>=MIN_TRADING_VALUE)&wok&(score>=70)&(rs>=rs80[day])&(gap<2.0)
        return base

    def report(tag, ret):
        n=len(ret)
        if n==0: print(f"    {tag}: 0건"); return
        r=ret*100; win=(r>0)
        wr=win.mean()*100; avg=r.mean(); med=np.median(r)
        aw=r[win].mean() if win.any() else 0; al=r[~win].mean() if (~win).any() else 0
        pf=abs(win.sum()*aw/((~win).sum()*al)) if (~win).any() and al!=0 else float('inf')
        print(f"    {tag}: {n:4d}건 | 승률 {wr:5.1f}% | 평균 {avg:+5.2f}% | 중앙 {med:+5.2f}% | 승{aw:+5.2f}/패{al:+6.2f} | PF {pf:4.2f}")

    policies=[("목표+5% / -4% / 10일",0.05,0.04),
              ("목표+7% / -4% / 10일",0.07,0.04),
              ("목표+10% / -4% / 10일",0.10,0.04),
              ("목표없음(시간+손절) / -4% / 10일",9.99,0.04)]
    print("\n"+"="*92)
    print("  실현손익 백테스트 (수수료·세금 미반영, 같은날 목표·손절시 손절우선=보수)")
    print("="*92)
    for name,tg,sp in policies:
        print(f"\n[{name}]")
        for V in ("V0","Q2"):
            report(f"{V}", simulate(hi[mask_of(V)],lo[mask_of(V)],clo[mask_of(V)], tg, sp))
    print("="*92)
    print("완료 / 총 done")
