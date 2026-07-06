# -*- coding: utf-8 -*-
"""
v14 히스토리 백필 — 지난 3개월을 라이브 봇과 동일한 로직으로 일별 시뮬레이션.
 결과: history.json (positions=현재 보유중, closed=3개월 청산기록, legacy 보존)
       candidates.json (마지막 날 기준 후보 = 오늘의 추천)
 규칙(라이브 봇과 동일): 코스피>120MA 국면 / 신고가-5%이내+거래대금30억+MA5>MA20
       진입=신호 다음날 시가(갭+2%↑ 취소) / 15거래일 만기 or -8% 손절 / 12슬롯
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import os, json, time, pickle
import numpy as np, pandas as pd
import types
# plotly 스텁 (fdr import 메모리 절약)
class _P:
    def __getattr__(s,k): return _p
    def __setattr__(s,k,v): pass
    def __call__(s,*a,**k): return _p
_p=_P()
_pl=types.ModuleType("plotly"); _pl.__path__=[]; _pl.__getattr__=lambda k:_p; sys.modules["plotly"]=_pl
for _s in ("io","graph_objects","express","subplots","offline","figure_factory"):
    _m=types.ModuleType("plotly."+_s); _m.__getattr__=lambda k:_p; setattr(_pl,_s,_m); sys.modules["plotly."+_s]=_m
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings("ignore")

SLOTS=12; HOLD_DAYS=15; STOP_LOSS=0.08; NEAR_HIGH=-5.0; GAP_MAX=2.0
MARKET_CAP_MIN=100_000_000_000; MARKET_CAP_MAX=5_000_000_000_000
MIN_TRADING_VALUE=3_000_000_000; REGIME_MA=120
SIM_MONTHS=3
PARTIAL="backfill_partial.pkl"

END=datetime.today().strftime("%Y-%m-%d")
SIM_START=(datetime.today()-timedelta(days=SIM_MONTHS*31)).strftime("%Y-%m-%d")
FETCH_START=(datetime.today()-timedelta(days=SIM_MONTHS*31+420)).strftime("%Y-%m-%d")

def build_data():
    t0=time.time()
    print("코스피 국면 데이터..."); sys.stdout.flush()
    ks=fdr.DataReader("KS11",FETCH_START,END)
    ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
    ksc=ks["Close"]; ma=ksc.rolling(REGIME_MA).mean()
    tdays_all=list(ks.index)
    regime={d:(bool(ksc[d]>ma[d]) if not np.isnan(ma[d]) else False) for d in tdays_all}
    sim_days=[d for d in tdays_all if d>=SIM_START]
    print(f"시뮬 기간 {sim_days[0]} ~ {sim_days[-1]} ({len(sim_days)}거래일, 국면ON {sum(regime[d] for d in sim_days)}일)")

    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb"))
        tickers=st["tickers"]; name_map=st["name_map"]; data=st["data"]; start=st["done"]
        print(f"[재개] {start}/{len(tickers)}"); sys.stdout.flush()
    else:
        alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
        filt=alls[(alls["Marcap"]>=MARKET_CAP_MIN)&(alls["Marcap"]<=MARKET_CAP_MAX)]
        tickers=filt["Code"].tolist(); name_map=dict(zip(filt["Code"],filt["Name"]))
        data={}; start=0
        print(f"종목 {len(tickers)}개 수집 시작"); sys.stdout.flush()

    for ii in range(start,len(tickers)):
        tk=tickers[ii]
        if ii%150==0 and ii>start:
            pickle.dump({"tickers":tickers,"name_map":name_map,"data":data,"done":ii},open(PARTIAL,"wb"))
            print(f"  [체크포인트] {ii}/{len(tickers)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,FETCH_START,END)
            if df.empty or len(df)<260: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
            close=df["Close"]; high=df["High"]
            h52=high.rolling(252).max()
            avgval=(df["Volume"]*close).rolling(20).mean()
            ma5=close.rolling(5).mean(); ma20=close.rolling(20).mean()
            data[tk]={
                "dates":list(df.index),
                "open":df["Open"].values.astype(np.float32),
                "close":close.values.astype(np.float32),
                "low":df["Low"].values.astype(np.float32),
                "h52":h52.values.astype(np.float32),
                "avgval":avgval.values.astype(np.float64),
                "ma5":ma5.values.astype(np.float32),
                "ma20":ma20.values.astype(np.float32),
            }
        except: continue
    pickle.dump({"tickers":tickers,"name_map":name_map,"data":data,"done":len(tickers)},open(PARTIAL,"wb"))
    print(f"수집 완료 {len(data)}종목 / {time.time()-t0:.0f}s")
    return sim_days, regime, name_map, data

def simulate(sim_days, regime, name_map, data):
    # 종목별 날짜→인덱스 맵
    dpos={tk:{d:i for i,d in enumerate(v["dates"])} for tk,v in data.items()}
    positions=[]   # {code,name,entry_date,entry_price,stop_price,...}
    closed=[]; daily_log=[]

    def candidates_on(day, held):
        out=[]
        for tk,v in data.items():
            if tk in held: continue
            i=dpos[tk].get(day)
            if i is None or i<252: continue
            c=v["close"][i]; h=v["h52"][i]; av=v["avgval"][i]; m5=v["ma5"][i]; m20=v["ma20"][i]
            if np.isnan(h) or h<=0 or np.isnan(av): continue
            d52=(c/h-1)*100
            if d52<NEAR_HIGH: continue
            if av<MIN_TRADING_VALUE: continue
            if not (m5>m20): continue
            out.append((tk,float(d52),float(c),float(av)))
        out.sort(key=lambda x:-x[1])
        return out

    pending=[]   # 어제 신호 → 오늘 시가 체결 대기 {code,ref_close,signal_date}
    for di,day in enumerate(sim_days):
        # 1) pending 체결 (오늘 시가)
        for pd_ in pending:
            tk=pd_["code"]; i=dpos[tk].get(day)
            if i is None: continue
            o=float(data[tk]["open"][i])
            gap=(o-pd_["ref_close"])/pd_["ref_close"]*100
            if gap>=GAP_MAX:
                continue    # 갭 취소
            positions.append({"code":tk,"name":name_map.get(tk,tk),
                "entry_date":day,"entry_price":round(o,2),
                "stop_price":round(o*(1-STOP_LOSS),2),"ref_close":pd_["ref_close"]})
        pending=[]

        # 2) 보유 갱신 (오늘 저가/종가 기준 손절·만기)
        kept=[]
        for p in positions:
            tk=p["code"]; i=dpos[tk].get(day)
            if i is None: kept.append(p); continue
            e=dpos[tk][p["entry_date"]]
            held_n=i-e+1
            lowmin=float(np.nanmin(data[tk]["low"][e:i+1]))
            cur=float(data[tk]["close"][i])
            if lowmin<=p["stop_price"]:
                closed.append({"code":tk,"name":p["name"],"entry_date":p["entry_date"],
                    "entry_price":p["entry_price"],"exit_date":day,
                    "exit_price":p["stop_price"],"ret_pct":round(-STOP_LOSS*100,2),"reason":"손절 -8%"})
            elif held_n>=HOLD_DAYS:
                ret=(cur-p["entry_price"])/p["entry_price"]*100
                closed.append({"code":tk,"name":p["name"],"entry_date":p["entry_date"],
                    "entry_price":p["entry_price"],"exit_date":day,
                    "exit_price":round(cur,2),"ret_pct":round(ret,2),"reason":f"{HOLD_DAYS}일 만기"})
            else:
                kept.append(p)
        positions=kept

        # 3) 신규 신호 (오늘 종가 기준 → 내일 시가 체결)
        if regime.get(day) and di<len(sim_days)-0:
            held={p["code"] for p in positions}|{pd_["code"] for pd_ in pending}
            empty=SLOTS-len(positions)
            if empty>0:
                cands=candidates_on(day,held)
                for tk,d52,c,av in cands[:empty]:
                    pending.append({"code":tk,"ref_close":c,"signal_date":day})
        daily_log.append((day,len(positions),len(closed)))
    return positions, closed, pending, dpos

if __name__=="__main__":
    sim_days, regime, name_map, data = build_data()
    positions, closed, pending, dpos = simulate(sim_days, regime, name_map, data)

    last=sim_days[-1]
    # 보유중 현황 필드 채우기
    for p in positions:
        tk=p["code"]; i=dpos[tk].get(last)
        if i is not None:
            e=dpos[tk][p["entry_date"]]
            p["current"]=round(float(data[tk]["close"][i]),2)
            p["ret_pct"]=round((p["current"]-p["entry_price"])/p["entry_price"]*100,2)
            p["days_held"]=i-e+1

    # 통계
    rets=[c["ret_pct"] for c in closed]
    wins=[r for r in rets if r>0]
    print(f"\n===== 3개월 백필 결과 ({sim_days[0]} ~ {last}) =====")
    print(f"청산 {len(closed)}건 | 승률 {len(wins)/len(rets)*100:.0f}% | 평균 {np.mean(rets):+.2f}% | 합산 {np.sum(rets):+.1f}%" if rets else "청산 0건")
    print(f"보유중 {len(positions)}개")
    for p in positions:
        print(f"  보유: {p['name']} {p['entry_date']} 진입 {p['entry_price']:,.0f} → 현재 {p.get('current',0):,.0f} ({p.get('ret_pct',0):+.1f}%, {p.get('days_held')}일)")

    # 오늘의 후보 (마지막 날 종가 기준 = 내일 시가 매수 대상)
    held={p["code"] for p in positions}
    # candidates_on는 simulate 내부함수라 재구현
    out=[]
    for tk,v in data.items():
        if tk in held: continue
        i=dpos[tk].get(last)
        if i is None or i<252: continue
        c=v["close"][i]; h=v["h52"][i]; av=v["avgval"][i]; m5=v["ma5"][i]; m20=v["ma20"][i]
        if np.isnan(h) or h<=0 or np.isnan(av): continue
        d52=(c/h-1)*100
        if d52<NEAR_HIGH or av<MIN_TRADING_VALUE or not (m5>m20): continue
        out.append({"code":tk,"name":name_map.get(tk,tk),"close":float(c),
                    "hi52":float(h),"d52":round(float(d52),2),
                    "avg_value_억":round(float(av)/100_000_000,1)})
    out.sort(key=lambda x:-x["d52"])
    print(f"\n오늘의 후보 {len(out)}개 (상위 10):")
    for c in out[:10]:
        print(f"  {c['name']}({c['code']}) 신고가 {c['d52']:+.2f}% / {c['avg_value_억']:,.0f}억 / {c['close']:,.0f}원")

    # history.json 병합 (legacy 보존)
    legacy=[]
    if os.path.exists("history.json"):
        with open("history.json",encoding="utf-8") as f:
            old=json.load(f)
        legacy = old if isinstance(old,list) else old.get("legacy",[])
    hist={"format":"v14","backfilled":f"{sim_days[0]}~{last}",
          "positions":positions,"closed":closed,"legacy":legacy}
    with open("history.json","w",encoding="utf-8") as f:
        json.dump(hist,f,ensure_ascii=False,indent=2)

    # candidates.json
    regime_msg=f"코스피>120일선 — 매매 ON (백필 {last} 기준)" if regime.get(last) else "약세장 — 현금 대기"
    with open("candidates.json","w",encoding="utf-8") as f:
        json.dump({"format":"v14","updated":datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "regime_on":bool(regime.get(last)),"regime_msg":regime_msg,
                   "candidates":out[:20],"new_entries":[]},f,ensure_ascii=False,indent=2)
    print("\nhistory.json / candidates.json 저장 완료")
    print("BACKFILL_DONE")
