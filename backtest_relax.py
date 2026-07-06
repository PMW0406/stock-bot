# -*- coding: utf-8 -*-
"""
v13.2 완화 실험 백테스트 (저메모리판)
- fdr가 끌어오는 무거운 plotly import를 스텁으로 우회 (MemoryError 방지)
- 후보행을 numpy 컬럼으로 저장(수 MB) → 다변형을 벡터화로 즉시 비교
- 1회 로드 후 backtest_relax_cache.npz 로 캐시 (재실행 즉시)
성공 정의: 보유 HOLD_DAYS 내 max_high가 +TARGET_GAIN 터치 (기존 백테스트 동일)
"""
import sys, io, time, os, types
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── plotly 스텁 (fdr import시 무거운 plotly 로드/검증 회피) ──
class _Perm:
    def __getattr__(self,k): return _perm
    def __setattr__(self,k,v): pass
    def __call__(self,*a,**kw): return _perm
    def __iter__(self): return iter(())
_perm=_Perm()
_plotly=types.ModuleType("plotly"); _plotly.__path__=[]
_plotly.__getattr__=lambda k: _perm
sys.modules["plotly"]=_plotly
for _sub in ("io","graph_objects","express","subplots","offline","figure_factory"):
    _mod=types.ModuleType("plotly."+_sub)
    _mod.__getattr__=lambda k: _perm
    setattr(_plotly,_sub,_mod)
    sys.modules["plotly."+_sub]=_mod

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta

CACHE_FILE       = "backtest_relax_cache.npz"
START_DATE       = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
START_DATE_STOCK = (datetime.today() - timedelta(days=365*2)).strftime("%Y-%m-%d")
END_DATE         = datetime.today().strftime("%Y-%m-%d")
MARKET_CAP_MIN   = 100_000_000_000
MARKET_CAP_MAX   = 5_000_000_000_000
MIN_TRADING_VALUE= 3_000_000_000
TARGET_GAIN      = 0.05
HOLD_DAYS        = 10

def market_states(trading_days, kospi_df, kosdaq_df):
    strict, mild = [], []
    for d in trading_days:
        s_ok, m_ok = True, True
        for mdf in (kospi_df, kosdaq_df):
            if d not in mdf.index: s_ok=m_ok=False; break
            idx=list(mdf.index).index(d)
            if idx<25: s_ok=m_ok=False; break
            close=mdf["Close"]; ma5=close.rolling(5).mean(); ma20=close.rolling(20).mean()
            c,m5,m20=float(close.iloc[idx]),float(ma5.iloc[idx]),float(ma20.iloc[idx])
            if c<m20: m_ok=False
            if c<m20 or m5<m20: s_ok=False
            if idx>=5:
                ret5=(c-float(close.iloc[idx-5]))/float(close.iloc[idx-5])
                if ret5<0: s_ok=False
        strict.append(s_ok); mild.append(m_ok)
    return np.array(strict), np.array(mild)

def build_weekly_cache(df):
    try:
        w=df.copy(); w.index=pd.to_datetime(w.index)
        wdf=w.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
        if len(wdf)<12: return {}
        cache={}; dates=list(df.index)
        wma10=wdf["Close"].rolling(10).mean()
        for ref_idx,d in enumerate(dates):
            if ref_idx<65: cache[d]=False; continue
            try:
                d_dt=pd.to_datetime(d); msk=wdf.index<=d_dt; nsl=int(msk.sum())
                if nsl<12: cache[d]=False; continue
                c=float(wdf["Close"].values[msk][-1]); m10=float(wma10.values[msk][-1])
                hs=wdf["High"].values[msk]; h52=float(hs[-53:].max()) if nsl>=53 else float(hs.max())
                ret13w=(float(df["Close"].iloc[ref_idx])-float(df["Close"].iloc[ref_idx-65]))/float(df["Close"].iloc[ref_idx-65])*100
                dd=(c-h52)/h52*100
                cache[d]=(not np.isnan(m10) and c>m10 and ret13w>0 and dd>=-40)
            except: cache[d]=False
        return cache
    except: return {}

def compute_metrics(df, ref_idx):
    if ref_idx<65: return None
    window=df.iloc[max(0,ref_idx-252):ref_idx+1]; today=df.iloc[ref_idx]
    close=window["Close"]; high=window["High"]; low=window["Low"]; volume=window["Volume"]
    ma5=close.rolling(5).mean(); ma20=close.rolling(20).mean(); ma60=close.rolling(60).mean()
    ma5n,ma20n,ma60n=float(ma5.iloc[-1]),float(ma20.iloc[-1]),float(ma60.iloc[-1])
    if any(np.isnan([ma5n,ma20n,ma60n])): return None
    score=0
    if today["Close"]>ma20n: score+=10
    if ma20n>ma60n: score+=10
    ma60p=float(ma60.iloc[-11]) if len(ma60)>11 else np.nan
    if not np.isnan(ma60p) and ma60n>ma60p: score+=10
    ret20=(today["Close"]-close.iloc[-21])/close.iloc[-21]*100 if len(close)>21 else 0
    if ret20>5: score+=20
    elif ret20>0: score+=10
    recent_high=close.iloc[-8:-1].max() if len(close)>=8 else close.max()
    pullback=(today["Close"]-recent_high)/recent_high*100
    if -8<=pullback<=-3: score+=20
    elif -3<pullback<=-1: score+=10
    vol_recent=volume.iloc[-4:-1].mean(); vol_before=volume.iloc[-9:-4].mean()
    vol_decrease=vol_recent<vol_before*0.9 if vol_before>0 else False
    vol_5avg=volume.iloc[-6:-1].mean(); vol_ratio=float(today["Volume"])/vol_5avg if vol_5avg>0 else 0
    if vol_decrease and vol_ratio>=1.5: score+=20
    elif vol_ratio>=1.5: score+=10
    prev_close=float(df.iloc[ref_idx-1]["Close"]); prev_ma5=float(ma5.iloc[-2]) if len(ma5)>=2 else np.nan
    if not np.isnan(prev_ma5) and prev_close<prev_ma5 and today["Close"]>ma5n: score+=10
    h_now=float(high.iloc[-1]); l_now=float(low.iloc[-1])
    cl=(today["Close"]-l_now)/(h_now-l_now) if (h_now-l_now)>0 else 0.5
    is_bearish=(today["Close"]<today["Open"]) and vol_ratio>=2.0
    avg_value=(volume.iloc[-21:-1]*close.iloc[-21:-1]).mean()
    return (bool(today["Close"]>ma20n), bool(ma20n>ma60n), float(pullback), float(vol_ratio),
            bool(vol_decrease), float(cl), bool(is_bearish), float(avg_value),
            float(ret20), int(score))

def build_cache():
    t0=time.time()
    print("시장 지수 수집..."); sys.stdout.flush()
    kospi=fdr.DataReader("KS11",START_DATE,END_DATE); kosdaq=fdr.DataReader("KQ11",START_DATE,END_DATE)
    kospi.index=pd.to_datetime(kospi.index).strftime("%Y-%m-%d"); kosdaq.index=pd.to_datetime(kosdaq.index).strftime("%Y-%m-%d")
    trading_days=sorted(kospi.index.tolist())
    day_pos={d:i for i,d in enumerate(trading_days)}
    mk_strict, mk_mild = market_states(trading_days,kospi,kosdaq)
    print(f"시장 양호일 strict={int(mk_strict.sum())} mild={int(mk_mild.sum())} / {len(trading_days)}일")

    alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
    filt=alls[(alls["Marcap"]>=MARKET_CAP_MIN)&(alls["Marcap"]<=MARKET_CAP_MAX)]
    tickers=filt["Code"].tolist()
    print(f"종목 {len(tickers)}개 / 2년치 로드+캐싱..."); sys.stdout.flush()

    prev_of={bd:(trading_days[i-1] if i>0 else None) for i,bd in enumerate(trading_days)}
    valid_buy=trading_days[:-HOLD_DAYS]
    # 컬럼 누적
    col_day=[]; col_gap=[]; col_gain=[]; col_wok=[]
    col_c20=[]; col_2060=[]; col_pb=[]; col_vr=[]; col_vd=[]; col_cl=[]; col_be=[]; col_av=[]; col_rs=[]; col_sc=[]
    rs_by_day={}
    n_load=0
    for i,tk in enumerate(tickers):
        if i%300==0: print(f"   {i}/{len(tickers)} ... ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,START_DATE_STOCK,END_DATE)
            if df.empty or len(df)<65: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
        except: continue
        n_load+=1
        dpos={d:j for j,d in enumerate(df.index)}
        wcache=build_weekly_cache(df)
        for bd in valid_buy:
            pv=prev_of.get(bd)
            if pv is None or pv not in dpos: continue
            m=compute_metrics(df,dpos[pv])
            if m is None: continue
            rs_by_day.setdefault(pv,{})[tk]=m[8]
            if bd not in dpos: continue
            buy=float(df.loc[bd,"Open"]); pc=float(df.loc[pv,"Close"])
            if buy<=0 or pc<=0: continue
            gap=(buy-pc)/pc*100
            fp=day_pos[bd]; fdays=trading_days[fp+1:fp+1+HOLD_DAYS]
            mh=max((float(df.loc[fd,"High"]) for fd in fdays if fd in dpos), default=0.0)
            if mh<=0: continue
            gain=(mh-buy)/buy
            c20,c2060,pb,vr,vd,cl,be,av,rs,sc=m
            col_day.append(day_pos[pv]); col_gap.append(gap); col_gain.append(gain); col_wok.append(bool(wcache.get(pv,False)))
            col_c20.append(c20); col_2060.append(c2060); col_pb.append(pb); col_vr.append(vr); col_vd.append(vd)
            col_cl.append(cl); col_be.append(be); col_av.append(av); col_rs.append(rs); col_sc.append(sc)
    print(f"   -> 로드 {n_load}개 / 후보행 {len(col_day)}개 / {time.time()-t0:.0f}s")

    nd=len(trading_days)
    rs80=np.full(nd,np.inf); rs70=np.full(nd,np.inf)
    for d,mp in rs_by_day.items():
        v=list(mp.values())
        if v: rs80[day_pos[d]]=np.percentile(v,80); rs70[day_pos[d]]=np.percentile(v,70)

    np.savez_compressed(CACHE_FILE,
        mk_strict=mk_strict, mk_mild=mk_mild, rs80=rs80, rs70=rs70, n_days=nd,
        day=np.array(col_day,np.int32), gap=np.array(col_gap,np.float32), gain=np.array(col_gain,np.float32),
        wok=np.array(col_wok,bool), c20=np.array(col_c20,bool), c2060=np.array(col_2060,bool),
        pb=np.array(col_pb,np.float32), vr=np.array(col_vr,np.float32), vd=np.array(col_vd,bool),
        cl=np.array(col_cl,np.float32), be=np.array(col_be,bool), av=np.array(col_av,np.float64),
        rs=np.array(col_rs,np.float32), sc=np.array(col_sc,np.int16))
    print(f"[캐시 저장] {CACHE_FILE}")

if __name__=="__main__":
    if not os.path.exists(CACHE_FILE):
        build_cache()
    Z=np.load(CACHE_FILE)
    mk={"strict":Z["mk_strict"],"mild":Z["mk_mild"]}; rsA={80:Z["rs80"],70:Z["rs70"]}
    day=Z["day"]; gap=Z["gap"]; gain=Z["gain"]; wok=Z["wok"]; c20=Z["c20"]; c2060=Z["c2060"]
    pb=Z["pb"]; vr=Z["vr"]; vd=Z["vd"]; cl=Z["cl"]; be=Z["be"]; av=Z["av"]; rs=Z["rs"]; sc=Z["sc"]
    print(f"시장 양호일 strict={int(mk['strict'].sum())} mild={int(mk['mild'].sum())} | 후보행 {len(day)}")

    def evaluate(name, *, market="strict", min_score=70, pbw=(-8,-0.5), vrw=(1.0,2.5),
                 need_vd=True, clw=(0.40,0.85), need_weekly=True, rs_pct=80):
        m = mk[market][day]
        mask = m & c20 & c2060
        mask &= (pb>=pbw[0]) & (pb<=pbw[1])
        mask &= (vr>=vrw[0]) & (vr<=vrw[1])
        if need_vd: mask &= vd
        mask &= ~be
        mask &= (cl>=clw[0]) & (cl<=clw[1])
        mask &= (av>=MIN_TRADING_VALUE)
        if need_weekly: mask &= wok
        score = sc + np.where(wok,5,0)
        mask &= (score>=min_score)
        thr = rsA[rs_pct][day]
        mask &= (rs>=thr)
        mask &= (gap<2.0)
        g=gain[mask]; n=len(g)
        if n==0:
            print(f"  {name:32s}: 0건"); return
        succ=g>=TARGET_GAIN; wr=succ.mean()*100; avg=g.mean()*100
        aw=g[succ].mean()*100 if succ.any() else 0.0
        al=g[~succ].mean()*100 if (~succ).any() else 0.0
        ev=(wr/100)*aw+(1-wr/100)*al; wk=n/(365/7)
        print(f"  {name:32s}: {n:5d}건 (주{wk:4.1f}) | 승률 {wr:5.1f}% | 평균 {avg:6.2f}% | 승{aw:5.2f}/패{al:6.2f} | 기대값 {ev:6.2f}%")

    print("\n"+"="*94)
    print(f"  변형 비교 (성공=보유{HOLD_DAYS}일내 +{int(TARGET_GAIN*100)}% 터치)")
    print("="*94)
    print(" [기준]");     evaluate("V0 baseline v13.2(현행)")
    print(" [단일완화]")
    evaluate("V1 볼륨창 0.8~3.0", vrw=(0.8,3.0))
    evaluate("V2 vol_decrease 해제", need_vd=False)
    evaluate("V3 점수 70->60", min_score=60)
    evaluate("V4 RS 20%->30%", rs_pct=70)
    evaluate("V5 풀백 -10~0", pbw=(-10,0))
    evaluate("V6 주봉 게이트해제", need_weekly=False)
    evaluate("V7 시장게이트 mild", market="mild")
    print(" [복합완화]")
    evaluate("C1 볼륨+점수60+RS30", vrw=(0.8,3.0), min_score=60, rs_pct=70)
    evaluate("C2 C1+vd해제", vrw=(0.8,3.0), min_score=60, rs_pct=70, need_vd=False)
    evaluate("C3 C2+주봉보너스화", vrw=(0.8,3.0), min_score=60, rs_pct=70, need_vd=False, need_weekly=False)
    evaluate("C4 C2+시장mild", market="mild", vrw=(0.8,3.0), min_score=60, rs_pct=70, need_vd=False)
    evaluate("C5 C4+주봉화+풀백-10~0", market="mild", vrw=(0.8,3.0), min_score=60, rs_pct=70,
             need_vd=False, need_weekly=False, pbw=(-10,0))
    print(" [품질보존형: 해로운 볼륨조건만 풀고 점수70·RS20 유지]")
    evaluate("Q1 볼륨0.8~3.0+vd해제",            vrw=(0.8,3.0), need_vd=False)
    evaluate("Q2 Q1+시장mild",         market="mild", vrw=(0.8,3.0), need_vd=False)
    evaluate("Q3 Q2+주봉해제",          market="mild", vrw=(0.8,3.0), need_vd=False, need_weekly=False)
    evaluate("Q4 Q2+풀백-10~0",        market="mild", vrw=(0.8,3.0), need_vd=False, pbw=(-10,0))
    print("="*94)
    print("완료 / 총 done")
