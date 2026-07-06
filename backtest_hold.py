# -*- coding: utf-8 -*-
"""
'오래 버티기' 실현손익 — Q2 신호를 대상으로 보유기간 × 손절폭 그리드.
 진입: 신호 다음날 시가.  청산 규칙:
   - (손절 켜짐) 보유중 저가 <= -손절% → 손절 청산
   - 그 외에는 목표 익절 없이 보유, 보유기간 마지막날 종가 청산 (승자 태우기)
 최대 60거래일(≈3개월) 경로를 캐시 → 여러 보유기간 즉시 비교.
 * 수수료/세금/슬리피지 미반영.
"""
import sys, os, time
import backtest_relax as B     # plotly 스텁 + 헬퍼 재사용
import numpy as np, pandas as pd

fdr=B.fdr; MTV=B.MIN_TRADING_VALUE
START_DATE=B.START_DATE; START_DATE_STOCK=B.START_DATE_STOCK; END_DATE=B.END_DATE
MCMIN=B.MARKET_CAP_MIN; MCMAX=B.MARKET_CAP_MAX
LOOKFWD=60
CACHE="backtest_hold_cache60.npz"

def build():
    t0=time.time(); print("시장 지수..."); sys.stdout.flush()
    kospi=fdr.DataReader("KS11",START_DATE,END_DATE); kosdaq=fdr.DataReader("KQ11",START_DATE,END_DATE)
    kospi.index=pd.to_datetime(kospi.index).strftime("%Y-%m-%d"); kosdaq.index=pd.to_datetime(kosdaq.index).strftime("%Y-%m-%d")
    trading_days=sorted(kospi.index.tolist()); day_pos={d:i for i,d in enumerate(trading_days)}
    mk_strict,mk_mild=B.market_states(trading_days,kospi,kosdaq)
    print(f"시장 strict={int(mk_strict.sum())} mild={int(mk_mild.sum())}/{len(trading_days)}")
    alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
    filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
    tickers=filt["Code"].tolist(); print(f"종목 {len(tickers)}개, 60일경로 수집..."); sys.stdout.flush()
    prev_of={bd:(trading_days[i-1] if i>0 else None) for i,bd in enumerate(trading_days)}
    valid_buy=trading_days[:-1]   # 최소 1일 앞 존재
    col=dict(day=[],gap=[],c20=[],c2060=[],pb=[],vr=[],cl=[],be=[],av=[],rs=[],sc=[],wok=[])
    lo=[]; clo=[]; rs_by_day={}; n=0
    for i,tk in enumerate(tickers):
        if i%300==0: print(f"  {i}/{len(tickers)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_DATE_STOCK,END_DATE)
            if df.empty or len(df)<65: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        n+=1; dpos={d:j for j,d in enumerate(df.index)}
        wc=B.build_weekly_cache(df)
        L=df["Low"].values; CL=df["Close"].values; O=df["Open"].values
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dpos: continue
            m=B.compute_metrics(df,dpos[pv])
            if m is None: continue
            rs_by_day.setdefault(pv,{})[tk]=m[8]
            if bd not in dpos: continue
            buy=float(O[dpos[bd]]); pc=float(CL[dpos[pv]])
            if buy<=0 or pc<=0: continue
            gap=(buy-pc)/pc*100
            fp=day_pos[bd]; fdays=trading_days[fp+1:fp+1+LOOKFWD]
            pl=np.full(LOOKFWD,np.nan,np.float32); pcx=pl.copy()
            for j,fd in enumerate(fdays):
                if fd in dpos: k=dpos[fd]; pl[j]=L[k]/buy; pcx[j]=CL[k]/buy
            if np.isnan(pcx).all(): continue
            c20,c2060,pb,vr,vd,cl,be,av,rs,sc=m
            col["day"].append(day_pos[pv]); col["gap"].append(gap); col["c20"].append(c20)
            col["c2060"].append(c2060); col["pb"].append(pb); col["vr"].append(vr); col["cl"].append(cl)
            col["be"].append(be); col["av"].append(av); col["rs"].append(rs); col["sc"].append(sc)
            col["wok"].append(bool(wc.get(pv,False)))
            lo.append(pl); clo.append(pcx)
    print(f"  -> 로드 {n} / 후보행 {len(col['day'])} / {time.time()-t0:.0f}s")
    nd=len(trading_days); rs80=np.full(nd,np.inf)
    for d,mp in rs_by_day.items():
        v=list(mp.values())
        if v: rs80[day_pos[d]]=np.percentile(v,80)
    np.savez_compressed(CACHE, mk_mild=mk_mild, rs80=rs80,
        day=np.array(col["day"],np.int32), gap=np.array(col["gap"],np.float32),
        c20=np.array(col["c20"],bool), c2060=np.array(col["c2060"],bool), pb=np.array(col["pb"],np.float32),
        vr=np.array(col["vr"],np.float32), cl=np.array(col["cl"],np.float32), be=np.array(col["be"],bool),
        av=np.array(col["av"],np.float64), rs=np.array(col["rs"],np.float32), sc=np.array(col["sc"],np.int16),
        wok=np.array(col["wok"],bool), lo=np.array(lo,np.float32), clo=np.array(clo,np.float32))
    print(f"[캐시 저장] {CACHE}")

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    Z=np.load(CACHE); mkm=Z["mk_mild"]; rs80=Z["rs80"]
    day=Z["day"]; gap=Z["gap"]; c20=Z["c20"]; c2060=Z["c2060"]; pb=Z["pb"]; vr=Z["vr"]
    cl=Z["cl"]; be=Z["be"]; av=Z["av"]; rs=Z["rs"]; sc=Z["sc"]; wok=Z["wok"]; lo=Z["lo"]; clo=Z["clo"]
    score=sc+np.where(wok,5,0)
    q2 = mkm[day]&c20&c2060&(pb>=-8)&(pb<=-0.5)&(vr>=0.8)&(vr<=3.0)&~be&(cl>=0.40)&(cl<=0.85)&(av>=MTV)&wok&(score>=70)&(rs>=rs80[day])&(gap<2.0)
    L=lo[q2]; C=clo[q2]; N=L.shape[0]
    print(f"Q2 신호 {N}건 (최대보유 {LOOKFWD}일 경로)\n")

    def sim(hold, stop):
        """목표없음(승자태우기). stop=None이면 무손절."""
        ret=np.full(N,np.nan)
        for i in range(N):
            lrow=L[i,:hold]; crow=C[i,:hold]
            valid=~np.isnan(crow)
            if not valid.any(): continue
            exited=False
            if stop is not None:
                stp=1-stop
                hit=np.where(valid & (lrow<=stp))[0]
                if len(hit): ret[i]=-stop; exited=True
            if not exited:
                cv=crow[valid]; ret[i]=cv[-1]-1.0
        return ret[~np.isnan(ret)]

    def line(tag, r):
        r=r*100; n=len(r); win=r>0
        wr=win.mean()*100; avg=r.mean(); med=np.median(r)
        aw=r[win].mean() if win.any() else 0; al=r[~win].mean() if (~win).any() else 0
        pf=abs(win.sum()*aw/((~win).sum()*al)) if (~win).any() and al!=0 else 99
        p90=np.percentile(r,90); worst=r.min()
        big=(r>=20).mean()*100
        print(f"  {tag:20s}| 승률{wr:5.1f}% 평균{avg:+6.2f}% 중앙{med:+6.2f}% PF{pf:5.2f} 90%값{p90:+6.1f} 최악{worst:+6.1f} +20%↑{big:4.1f}%")

    # 최대 한 달(약 20거래일)까지만
    for stop in (None,0.08,0.10,0.15):
        stag = "무손절" if stop is None else f"-{int(stop*100)}%손절"
        print(f"[{stag}]  (목표 익절 없음 · 보유기간말 종가청산)")
        for hold in (5,10,15,20):
            line(f"보유 {hold}일", sim(hold,stop))
        print()
    print("="*100); print("완료 / 총 done")
