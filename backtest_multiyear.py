# -*- coding: utf-8 -*-
"""
다년(5년) 백테스트 — 연도별 분해. '2026 비정상 강세장 착시'인지 검증.
 실제 Q2 신호 + 승자태우기 청산 + 포트폴리오 복리(K종목 분산, 수수료 0.3%).
 메모리 절약: RS 임계값은 전 종목 ret20로 계산하되, 경로(path)는 preQ2 통과행만 저장.
 ⚠️ 현재 상장종목만 대상 → 상장폐지 종목 빠져 상방편향(생존편향) 있음.
"""
import sys, os, time, pickle
import backtest_relax as B      # plotly 스텁 + fdr + helpers
import numpy as np, pandas as pd
PARTIAL="backtest_multiyear_partial.pkl"

fdr=B.fdr; MTV=B.MIN_TRADING_VALUE
from datetime import datetime, timedelta
END=datetime.today().strftime("%Y-%m-%d")
START_MKT=(datetime.today()-timedelta(days=365*5)).strftime("%Y-%m-%d")
START_STK=(datetime.today()-timedelta(days=365*5+260)).strftime("%Y-%m-%d")
MCMIN=B.MARKET_CAP_MIN; MCMAX=B.MARKET_CAP_MAX
LOOKFWD=15; COST=0.003
CACHE="backtest_multiyear_cache.npz"

def build():
    t0=time.time(); print("시장 지수 5년..."); sys.stdout.flush()
    kospi=fdr.DataReader("KS11",START_MKT,END); kosdaq=fdr.DataReader("KQ11",START_MKT,END)
    kospi.index=pd.to_datetime(kospi.index).strftime("%Y-%m-%d"); kosdaq.index=pd.to_datetime(kosdaq.index).strftime("%Y-%m-%d")
    tdays=sorted(kospi.index.tolist()); dpm={d:i for i,d in enumerate(tdays)}
    mk_s,mk_m=B.market_states(tdays,kospi,kosdaq)
    print(f"거래일 {len(tdays)} ({tdays[0]}~{tdays[-1]}) | 시장 strict={int(mk_s.sum())} mild={int(mk_m.sum())}")
    prev_of={bd:(tdays[i-1] if i>0 else None) for i,bd in enumerate(tdays)}
    valid_buy=tdays[:-1]
    # ── 체크포인트 재개 ──
    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb"))
        tickers=st["tickers"]; col=st["col"]; lo=st["lo"]; clo=st["clo"]; rs_by_day=st["rs_by_day"]; start=st["done"]
        print(f"[재개] {start}/{len(tickers)} 부터 (누적행 {len(col['day'])})"); sys.stdout.flush()
    else:
        alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
        filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
        tickers=filt["Code"].tolist()
        col=dict(day=[],gap=[],ret20=[],score=[]); lo=[]; clo=[]; rs_by_day={}; start=0
        print(f"종목 {len(tickers)}개, 5년 스캔..."); sys.stdout.flush()
    n=0
    for ii in range(start,len(tickers)):
        tk=tickers[ii]
        if ii%150==0 and ii>start:
            pickle.dump({"tickers":tickers,"col":col,"lo":lo,"clo":clo,"rs_by_day":rs_by_day,"done":ii},
                        open(PARTIAL,"wb"))
            print(f"  [체크포인트] {ii}/{len(tickers)} ({time.time()-t0:.0f}s, 행{len(col['day'])})"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_STK,END)
            if df.empty or len(df)<130: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        n+=1; dp={d:j for j,d in enumerate(df.index)}
        wc=B.build_weekly_cache(df)
        Ov=df["Open"].values; Cv=df["Close"].values; Lv=df["Low"].values
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dp: continue
            r=dp[pv]
            m=B.compute_metrics(df,r)
            if m is None: continue
            c20,c2060,pb,vr,vd,cl,be,av,ret20,sc=m
            rs_by_day.setdefault(pv,{})[tk]=ret20
            wok=bool(wc.get(pv,False)); score=sc+(5 if wok else 0)
            di=dpm[pv]
            preq2 = (mk_m[di] and c20 and c2060 and (-8<=pb<=-0.5) and (0.8<=vr<=3.0)
                     and (not be) and (0.40<=cl<=0.85) and (av>=MTV) and wok and score>=70)
            if not preq2: continue
            if bd not in dp: continue
            buy=float(Ov[dp[bd]]); pc=float(Cv[r])
            if buy<=0 or pc<=0: continue
            if (buy-pc)/pc*100>=2.0: continue
            fp=dpm[bd]; fdays=tdays[fp+1:fp+1+LOOKFWD]
            pl=np.full(LOOKFWD,np.nan,np.float32); pc2=pl.copy()
            for j,fd in enumerate(fdays):
                if fd in dp: k=dp[fd]; pl[j]=Lv[k]/buy; pc2[j]=Cv[k]/buy
            if np.isnan(pc2).all(): continue
            col["day"].append(di); col["gap"].append((buy-pc)/pc*100)
            col["ret20"].append(ret20); col["score"].append(score); lo.append(pl); clo.append(pc2)
    print(f"  -> 로드 {n} / preQ2행 {len(col['day'])} / {time.time()-t0:.0f}s")
    nd=len(tdays); rs80=np.full(nd,np.inf)
    for d,mp in rs_by_day.items():
        v=list(mp.values())
        if v: rs80[dpm[d]]=np.percentile(v,80)
    np.savez_compressed(CACHE, tdays=np.array(tdays), mk_mild=mk_m, rs80=rs80,
        day=np.array(col["day"],np.int32), gap=np.array(col["gap"],np.float32),
        ret20=np.array(col["ret20"],np.float32), score=np.array(col["score"],np.int16),
        lo=np.array(lo,np.float32), clo=np.array(clo,np.float32))
    if os.path.exists(PARTIAL): os.remove(PARTIAL)
    print(f"[캐시 저장] {CACHE}")

def outcomes(day,ret20,score,rs80,lo,clo, stop, hold):
    """Q2 최종(RS컷 포함) 신호의 (entry_idx, exit_idx, ret_net, score)."""
    fin = ret20>=rs80[day]
    out=[]
    for i in np.where(fin)[0]:
        L=lo[i,:hold]; C=clo[i,:hold]; v=~np.isnan(C)
        if not v.any(): continue
        if stop is not None:
            sp=1-stop; hit=np.where(v&(L<=sp))[0]
            if len(hit): off=int(hit[0]); r=-stop
            else: last=int(np.where(v)[0][-1]); off=last; r=float(C[last]-1)
        else:
            last=int(np.where(v)[0][-1]); off=last; r=float(C[last]-1)
        d=int(day[i]); out.append((d+1, d+2+off, r-COST, float(score[i])))
    return out

def run_portfolio(trades, K, ndays, tdays):
    by_entry={}
    for e,x,r,s in trades: by_entry.setdefault(e,[]).append((s,x,r))
    cash=1.0; pos=[]; eq_series=np.empty(ndays+LOOKFWD+3)
    for t in range(ndays+LOOKFWD+2):
        pos=[(x,sz,r) for (x,sz,r) in pos if x!=t] if False else pos
        keep=[]
        for x,sz,r in pos:
            if x==t: cash+=sz*(1+r)
            else: keep.append((x,sz,r))
        pos=keep
        eq=cash+sum(s for _,s,_ in pos); eq_series[t]=eq
        for s,x,r in sorted(by_entry.get(t,[]),key=lambda z:z[0],reverse=True):
            if len(pos)>=K: break
            size=eq/K
            if 0<size<=cash+1e-12: cash-=size; pos.append((x,size,r))
    final=cash+sum(s for _,s,_ in pos)
    eqs=eq_series[:ndays]
    peak=np.maximum.accumulate(eqs); mdd=((eqs-peak)/peak).min()*100
    # 연도별 수익률
    years={}
    yr_of=[d[:4] for d in tdays]
    # 각 연도 마지막 거래일 인덱스
    last_idx={}
    for i,y in enumerate(yr_of): last_idx[y]=i
    ys=sorted(set(yr_of)); yret={}
    prev_eq=1.0; prev_i=0
    for y in ys:
        li=last_idx[y]; e_end=eqs[li]
        yret[y]=(e_end/prev_eq-1)*100; prev_eq=e_end
    return final-1, mdd, yret, eqs

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    Z=np.load(CACHE, allow_pickle=True)
    tdays=list(Z["tdays"]); mk_m=Z["mk_mild"]; rs80=Z["rs80"]
    day=Z["day"]; gap=Z["gap"]; ret20=Z["ret20"]; score=Z["score"]; lo=Z["lo"]; clo=Z["clo"]
    ndays=len(tdays)
    yr_of=[d[:4] for d in tdays]; ys=sorted(set(yr_of))
    # 연도별 신호수/시장개방일
    print(f"기간 {tdays[0]}~{tdays[-1]} / preQ2행 {len(day)}")
    finmask = ret20>=rs80[day]
    print("\n[연도별] 시장개방일(mild) / Q2최종신호수")
    for y in ys:
        di=[i for i,d in enumerate(tdays) if d[:4]==y]
        opendays=int(mk_m[di].sum())
        sigs=int(((day>=di[0])&(day<=di[-1])&finmask).sum()) if di else 0
        print(f"  {y}: 개방 {opendays:3d}일 / Q2신호 {sigs:4d}건")

    print("\n[포트폴리오] Q2 + 승자태우기, 수수료0.3% — 연도별 수익률")
    for stop,hold,K in [(0.20,10,8),(None,10,8),(0.15,10,8),(0.20,10,5)]:
        tr=outcomes(day,ret20,score,rs80,lo,clo,stop,hold)
        tot,mdd,yret,_=run_portfolio(tr,K,ndays,tdays)
        stag="무손절" if stop is None else f"-{int(stop*100)}%"
        yrs_str=" ".join(f"{y}:{yret[y]:+5.1f}%" for y in ys)
        cagr=((1+tot)**(1/ (ndays/245))-1)*100
        print(f"\n  [{stag}/{hold}일/K{K}] 5년총 {tot*100:+.1f}% · CAGR {cagr:+.1f}% · MDD {mdd:.1f}%")
        print(f"     연도별: {yrs_str}")
    print("\n완료 / 총 done")
