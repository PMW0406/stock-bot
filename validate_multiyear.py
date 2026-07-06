# -*- coding: utf-8 -*-
"""
결정적 검증: 5년 연도별로 'Q2 선정 vs 강세장 무작위' 10일수익 차이.
 엣지가 나쁜해(2022·2024)에도 +면 진짜, 강세장에만 +면 가짜.
 강세장 = 코스피 120일선 위. 매수=신호 다음날 시가, 매도=10거래일 후 종가.
 메모리 절약: 강세장ON 행만 (day,ret10,ret20,q2pre,score) 저장 → RS는 사후 적용.
 중간저장(체크포인트)으로 세션 끊겨도 재개.
"""
import sys, os, time, pickle
import backtest_relax as B
import numpy as np, pandas as pd
from datetime import datetime, timedelta
fdr=B.fdr; MTV=B.MIN_TRADING_VALUE
END=datetime.today().strftime("%Y-%m-%d")
START_MKT=(datetime.today()-timedelta(days=365*5)).strftime("%Y-%m-%d")
START_STK=(datetime.today()-timedelta(days=365*5+260)).strftime("%Y-%m-%d")
MCMIN=B.MARKET_CAP_MIN; MCMAX=B.MARKET_CAP_MAX
HOLD=10
PARTIAL="validate_partial.pkl"; CACHE="validate_cache.npz"

def build():
    t0=time.time(); print("지수 5년..."); sys.stdout.flush()
    ks=fdr.DataReader("KS11",START_MKT,END); ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
    tdays=sorted(ks.index.tolist()); dpm={d:i for i,d in enumerate(tdays)}
    ksc=ks["Close"].reindex(tdays).ffill(); reg=(ksc>ksc.rolling(120).mean()).fillna(False).values
    print(f"거래일 {len(tdays)} {tdays[0]}~{tdays[-1]} / 강세장(120MA위) {int(reg.sum())}일")
    prev_of={bd:(tdays[i-1] if i>0 else None) for i,bd in enumerate(tdays)}
    valid_buy=tdays[:-(HOLD+1)]
    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb"))
        tickers=st["tickers"]; col=st["col"]; rs_by_day=st["rs_by_day"]; start=st["done"]
        print(f"[재개] {start}/{len(tickers)} (행{len(col['day'])})"); sys.stdout.flush()
    else:
        alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
        filt=alls[(alls["Marcap"]>=MCMIN)&(alls["Marcap"]<=MCMAX)]
        tickers=filt["Code"].tolist()
        col=dict(day=[],ret10=[],ret20=[],q2pre=[],score=[]); rs_by_day={}; start=0
        print(f"종목 {len(tickers)}개, 5년 스캔..."); sys.stdout.flush()
    for ii in range(start,len(tickers)):
        tk=tickers[ii]
        if ii%150==0 and ii>start:
            pickle.dump({"tickers":tickers,"col":col,"rs_by_day":rs_by_day,"done":ii},open(PARTIAL,"wb"))
            print(f"  [체크포인트] {ii}/{len(tickers)} ({time.time()-t0:.0f}s, 행{len(col['day'])})"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_STK,END)
            if df.empty or len(df)<130: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        dp={d:j for j,d in enumerate(df.index)}; wc=B.build_weekly_cache(df)
        Ov=df["Open"].values; Cv=df["Close"].values
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dp: continue
            r=dp[pv]
            m=B.compute_metrics(df,r)
            if m is None: continue
            c20,c2060,pb,vr,vd,cl,be,av,ret20,sc=m
            rs_by_day.setdefault(pv,{})[tk]=ret20
            di=dpm[pv]
            if not reg[di]: continue           # 강세장 아니면 스킵(양 그룹 다 강세장만)
            if bd not in dp: continue
            fp=dpm[bd]
            if fp+HOLD>=len(tdays): continue
            sell_day=tdays[fp+HOLD]
            if sell_day not in dp: continue
            buy=float(Ov[dp[bd]]); sell=float(Cv[dp[sell_day]])
            if buy<=0: continue
            wok=bool(wc.get(pv,False)); score=sc+(5 if wok else 0)
            q2pre=(c20 and c2060 and (-8<=pb<=-0.5) and (0.8<=vr<=3.0) and (not be)
                   and (0.40<=cl<=0.85) and (av>=MTV) and wok and score>=70)
            col["day"].append(di); col["ret10"].append((sell-buy)/buy*100)
            col["ret20"].append(ret20); col["q2pre"].append(q2pre); col["score"].append(score)
    print(f"  -> 후보행 {len(col['day'])} / {time.time()-t0:.0f}s")
    nd=len(tdays); rs80=np.full(nd,np.inf)
    for d,mp in rs_by_day.items():
        v=list(mp.values())
        if v: rs80[dpm[d]]=np.percentile(v,80)
    np.savez_compressed(CACHE, tdays=np.array(tdays),
        day=np.array(col["day"],np.int32), ret10=np.array(col["ret10"],np.float32),
        ret20=np.array(col["ret20"],np.float32), q2pre=np.array(col["q2pre"],bool),
        score=np.array(col["score"],np.int16), rs80=rs80)
    if os.path.exists(PARTIAL): os.remove(PARTIAL)
    print(f"[캐시 저장] {CACHE}")

if __name__=="__main__":
    if not os.path.exists(CACHE): build()
    Z=np.load(CACHE, allow_pickle=True)
    tdays=[d.decode() if isinstance(d,bytes) else str(d) for d in Z["tdays"]]
    day=Z["day"]; ret10=Z["ret10"]; ret20=Z["ret20"]; q2pre=Z["q2pre"]; score=Z["score"]; rs80=Z["rs80"]
    yr_of=np.array([tdays[d][:4] for d in day])
    Q2 = q2pre & (ret20>=rs80[day])
    print("\n[연도별] 강세장(120MA위)에서 Q2선정 vs 무작위 10일수익")
    print(f"  {'연도':>5} | {'Q2평균':>7} {'무작위평균':>9} {'엣지(Q2-무작위)':>13} | {'Q2표본':>7}")
    for y in sorted(set(yr_of)):
        m=yr_of==y
        q=ret10[m & Q2]; rnd=ret10[m]
        if len(q)==0:
            print(f"  {y:>5} | Q2신호 0건"); continue
        edge=q.mean()-rnd.mean()
        flag="✅" if edge>0 else "❌"
        print(f"  {y:>5} | {q.mean():+6.2f}% {rnd.mean():+8.2f}% {edge:+12.2f}%p {flag} | {len(q):7d}")
    # 전체
    q=ret10[Q2]; rnd=ret10
    print(f"  {'전체':>5} | {q.mean():+6.2f}% {rnd.mean():+8.2f}% {q.mean()-rnd.mean():+12.2f}%p    | {len(q):7d}")
    print("\n완료 / 총 done")
