# -*- coding: utf-8 -*-
"""
v14.8 히스토리 백필 — 현행 전략 전체 규칙으로 3개월 재시뮬레이션
 A트랙: 신선(5일)신고가 -5~-1% + 이격4~8 + ret20 10~25 + 폭증제외 + 실적필터(역성장 제외/GOOD 1.5x)
        15일 or 종가 -10% 손절 · 12슬롯 · 갭 ±취소 · 서킷브레이커(10중7→5일)
 B트랙: 초대형 RSI2<10 + >MA200 + 외인20일수급(강매도 제외·매집순) · +3% 고가터치 or 10일 · 3슬롯
 국면: 코스피 120MA + 5일 히스테리시스
 산출: history.json (positions/closed/breaker, legacy 보존)
"""
import sys, io, os, time, pickle, re, json
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import types
class _P:
    def __getattr__(s,k): return _p
    def __setattr__(s,k,v): pass
    def __call__(s,*a,**k): return _p
_p=_P()
pl=types.ModuleType("plotly"); pl.__path__=[]; pl.__getattr__=lambda k:_p; sys.modules["plotly"]=pl
for s in ("io","graph_objects","express","subplots","offline","figure_factory"):
    m=types.ModuleType("plotly."+s); m.__getattr__=lambda k:_p; setattr(pl,s,m); sys.modules["plotly."+s]=m
import numpy as np, pandas as pd, FinanceDataReader as fdr
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings("ignore")

SIM_START="2026-04-06"
FETCH_START=(datetime.today()-timedelta(days=460)).strftime("%Y-%m-%d")
END=datetime.today().strftime("%Y-%m-%d")
PARTIAL="bf148_partial.pkl"

# ── 실적 타임라인 (시점 정합) ──
def build_earnings():
    FIN=pd.read_parquet("dart_fin.parquet"); EV=pd.read_parquet("dart_events.parquet")
    pat=re.compile(r"(사업|반기|분기)보고서.*?\((\d{4})\.(\d{2})\)")
    recs={}
    for _,e in EV.iterrows():
        m=pat.search(e.report_nm)
        if not m: continue
        rc={"03":"11013","06":"11012","09":"11014","12":"11011"}.get(m.group(3))
        if m.group(1)=="사업": rc="11011"
        if rc is None: continue
        key=(e.stock_code,m.group(2),rc)
        if key not in recs or e.rcept_dt<recs[key]: recs[key]=e.rcept_dt
    FIN=FIN[FIN.account=="영업이익"].copy(); FIN["pri"]=(FIN.fs_div!="CFS").astype(int)
    FIN=FIN.sort_values("pri").drop_duplicates(subset=["stock_code","bsns_year","reprt_code"])
    tl={}
    for _,f in FIN.iterrows():
        rd=recs.get((f.stock_code,f.bsns_year,f.reprt_code))
        if rd is None or f.thstrm is None: continue
        dt=f"{rd[:4]}-{rd[4:6]}-{rd[6:]}"
        stt=None; yoy=None
        if f.frmtrm is not None:
            if f.frmtrm>0 and f.thstrm>0: yoy=(f.thstrm/f.frmtrm-1)*100; stt="G"
            elif f.frmtrm<=0<f.thstrm: stt="TURN"
            elif f.thstrm<=0: stt="LOSS"
        else: stt="G" if f.thstrm>0 else "LOSS"
        tl.setdefault(f.stock_code,[]).append((dt,stt,yoy))
    for tk in tl: tl[tk].sort()
    return tl
def earn_state(tl,tk,date):
    t=tl.get(tk)
    if not t: return "NEUTRAL",None
    best=(None,None)
    for dt,stt,yoy in t:
        if dt<=date: best=(stt,yoy)
        else: break
    stt,yoy=best
    if stt=="G" and yoy is not None and yoy<-10: return "BAD",yoy
    if stt=="TURN" or (stt=="G" and yoy is not None and yoy>=50): return "GOOD",yoy
    return "NEUTRAL",yoy

def rsi2(close):
    d=pd.Series(close).diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ru=up.ewm(alpha=1/2,adjust=False).mean(); rd=dn.ewm(alpha=1/2,adjust=False).mean()
    return (100-100/(1+ru/rd.replace(0,np.nan))).fillna(50).values

def build():
    t0=time.time()
    ks=fdr.DataReader("KS11",FETCH_START,END); ks.index=pd.to_datetime(ks.index).strftime("%Y-%m-%d")
    tdays=sorted(ks.index.tolist())
    ksc=ks["Close"]; ma=ksc.rolling(120).mean()
    rawr=(ksc>ma).fillna(False).tolist()
    reg={}; st=rawr[0]; cnt=0
    for d,x in zip(tdays,rawr):
        if x!=st:
            cnt+=1
            if cnt>=5: st=x; cnt=0
        else: cnt=0
        reg[d]=st
    sim_days=[d for d in tdays if d>=SIM_START]
    print(f"시뮬 {sim_days[0]}~{sim_days[-1]} ({len(sim_days)}일, 국면ON {sum(reg[d] for d in sim_days)}일)")
    uni=pd.read_csv("universe_cache.csv",dtype={"Code":str})
    A_tks=uni[(uni.Marcap>=1e11)&(uni.Marcap<=5e12)]["Code"].tolist()
    B_tks=uni[uni.Marcap>=5e12]["Code"].tolist()
    name_map=dict(zip(uni.Code,uni.Name))
    if os.path.exists(PARTIAL):
        stt=pickle.load(open(PARTIAL,"rb")); data=stt["data"]; done=stt["done"]
        print(f"[재개] {done}/{len(A_tks)+len(B_tks)}")
    else:
        data={}; done=0
    alltk=A_tks+B_tks
    for i in range(done,len(alltk)):
        tk=alltk[i]
        if i%150==0 and i>done:
            pickle.dump({"data":data,"done":i},open(PARTIAL,"wb"))
            print(f"  [ckpt] {i}/{len(alltk)} ({time.time()-t0:.0f}s)"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,FETCH_START,END)
            if df.empty or len(df)<260: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
            data[tk]={"dates":list(df.index),
                "O":df["Open"].values.astype(float),"H":df["High"].values.astype(float),
                "L":df["Low"].values.astype(float),"C":df["Close"].values.astype(float),
                "V":df["Volume"].values.astype(float)}
        except: pass
    pickle.dump({"data":data,"done":len(alltk)},open(PARTIAL,"wb"))
    print(f"수집 {len(data)}종목 / {time.time()-t0:.0f}s")
    return tdays, sim_days, reg, data, set(A_tks), set(B_tks), name_map

def indicators(v):
    C=v["C"]; H=v["H"]; V=v["V"]
    s=pd.Series(C)
    return dict(ma5=s.rolling(5).mean().values, ma20=s.rolling(20).mean().values,
        ma200=s.rolling(200).mean().values,
        h52=pd.Series(H).rolling(252).max().values,
        aval=pd.Series(C*V).rolling(20).mean().values,
        vol5=pd.Series(V).rolling(5).mean().shift(1).values,
        fresh=pd.Series(H).rolling(252).apply(lambda x:251-int(np.argmax(x.values)),raw=False).values,
        r2=rsi2(C))

def flow20_at(FLOW, tk, date, aval):
    g=FLOW.get(tk)
    if g is None or not aval or aval<=0: return None
    sub=[x for x in g if x[0]<=date][-20:]
    if len(sub)<18: return None
    return sum(x[1] for x in sub)/(aval*20)*100

if __name__=="__main__":
    tdays, sim_days, reg, data, A_set, B_set, name_map = build()
    TL=build_earnings()
    FL=pd.read_parquet("flows_mega.parquet")
    FL["date"]=pd.to_datetime(FL["date"]).dt.strftime("%Y-%m-%d")
    FLOW={}
    for tk,g in FL.groupby("ticker"):
        # (date, 순매수대금) — 종가는 가격데이터에서
        v=data.get(tk)
        if v is None: continue
        dpos={d:i for i,d in enumerate(v["dates"])}
        lst=[]
        for _,row in g.iterrows():
            i=dpos.get(row["date"])
            if i is not None: lst.append((row["date"], row["frgn"]*v["C"][i]))
        lst.sort(); FLOW[tk]=lst
    IND={tk:indicators(v) for tk,v in data.items()}
    dpos_all={tk:{d:i for i,d in enumerate(v["dates"])} for tk,v in data.items()}
    prev_of={}
    for i,d in enumerate(tdays):
        prev_of[d]=tdays[i-1] if i>0 else None

    positions=[]; closed=[]; pendA=[]; pendB=[]
    recent=[]; pause=0
    for day in sim_days:
        # 0) pending 체결
        for p in pendA:
            tk=p["code"]; i=dpos_all[tk].get(day)
            if i is None: continue
            o=data[tk]["O"][i]; gap=(o-p["ref"])/p["ref"]*100
            if gap>=2 or gap<=-3: continue
            positions.append(dict(code=tk,name=name_map.get(tk,tk),track="A",
                entry_date=day,entry_price=round(float(o),2),
                stop_price=round(float(o)*0.90,2),good=p["good"]))
        pendA=[]
        for p in pendB:
            tk=p["code"]; i=dpos_all[tk].get(day)
            if i is None: continue
            o=data[tk]["O"][i]
            positions.append(dict(code=tk,name=name_map.get(tk,tk),track="B",
                entry_date=day,entry_price=round(float(o),2),
                target_price=round(float(o)*1.03,2)))
        pendB=[]
        # 1) 보유 갱신
        kept=[]
        for p in positions:
            tk=p["code"]; i=dpos_all[tk].get(day)
            if i is None: kept.append(p); continue
            e=dpos_all[tk][p["entry_date"]]
            held=i-e+1; C=data[tk]["C"]; H=data[tk]["H"]
            if p["track"]=="A":
                if C[i]<=p["stop_price"]:
                    ret=(C[i]-p["entry_price"])/p["entry_price"]*100
                    closed.append(dict(code=tk,name=p["name"],track="A",entry_date=p["entry_date"],
                        entry_price=p["entry_price"],exit_date=day,exit_price=round(float(C[i]),2),
                        ret_pct=round(ret,2),reason="손절 -10%(종가)")); recent.append(True); continue
                if held>=15:
                    ret=(C[i]-p["entry_price"])/p["entry_price"]*100
                    closed.append(dict(code=tk,name=p["name"],track="A",entry_date=p["entry_date"],
                        entry_price=p["entry_price"],exit_date=day,exit_price=round(float(C[i]),2),
                        ret_pct=round(ret,2),reason="15일 만기")); recent.append(False); continue
            else:
                if H[i]>=p["target_price"]:
                    closed.append(dict(code=tk,name=p["name"],track="B",entry_date=p["entry_date"],
                        entry_price=p["entry_price"],exit_date=day,exit_price=p["target_price"],
                        ret_pct=3.0,reason="목표 +3%")); continue
                if held>=10:
                    ret=(C[i]-p["entry_price"])/p["entry_price"]*100
                    closed.append(dict(code=tk,name=p["name"],track="B",entry_date=p["entry_date"],
                        entry_price=p["entry_price"],exit_date=day,exit_price=round(float(C[i]),2),
                        ret_pct=round(ret,2),reason="10일 만기(B)")); continue
            kept.append(p)
        positions=kept
        # 2) 브레이커 (A 손절 기준)
        if pause==0 and len(recent)>=10 and sum(recent[-10:])>=7:
            pause=5; recent=[]
        breaker_on = pause>0
        if pause>0: pause-=1
        # 3) 신호 (당일 종가 기준 → 익일 체결)
        if not reg.get(day): continue
        held_codes={p["code"] for p in positions}
        nA=sum(1 for p in positions if p["track"]=="A")
        nB=sum(1 for p in positions if p["track"]=="B")
        if not breaker_on and nA<12:
            cands=[]
            for tk in A_set:
                if tk in held_codes or tk not in data: continue
                v=data[tk]; ind=IND[tk]; i=dpos_all[tk].get(day)
                if i is None or i<252: continue
                C=v["C"]; V=v["V"]
                h52=ind["h52"][i]; aval=ind["aval"][i]
                if not h52 or np.isnan(aval) or aval<3e9: continue
                d52=(C[i]/h52-1)*100
                if not (-5<=d52<-1): continue
                ma5,ma20=ind["ma5"][i],ind["ma20"][i]
                if np.isnan(ma20) or ma20<=0: continue
                prem=(ma5/ma20-1)*100
                if not (4<=prem<8): continue
                ret20=(C[i]-C[i-20])/C[i-20]*100
                if not (10<=ret20<25): continue
                v5=ind["vol5"][i]
                if v5 and v5>0 and V[i]/v5>=4: continue
                if ind["fresh"][i]>5: continue
                est,yy=earn_state(TL,tk,day)
                if est=="BAD": continue
                cands.append((0 if est=="GOOD" else 1, -d52, tk, C[i], est=="GOOD"))
            cands.sort()
            for _,_,tk,ref,good in cands[:max(0,12-nA)]:
                pendA.append(dict(code=tk,ref=float(ref),good=good))
        if nB<3:
            candsB=[]
            for tk in B_set:
                if tk in held_codes or tk not in data: continue
                v=data[tk]; ind=IND[tk]; i=dpos_all[tk].get(day)
                if i is None or i<200: continue
                C=v["C"]
                if np.isnan(ind["ma200"][i]) or C[i]<=ind["ma200"][i]: continue
                if ind["r2"][i]>=10: continue
                f20=flow20_at(FLOW,tk,day,ind["aval"][i])
                if f20 is not None and f20<=-10: continue
                candsB.append((-(f20 if f20 is not None else -99), tk))
            candsB.sort()
            for _,tk in candsB[:max(0,3-nB)]:
                pendB.append(dict(code=tk))
    # 현황 필드
    last=sim_days[-1]
    for p in positions:
        tk=p["code"]; i=dpos_all[tk].get(last)
        if i is not None:
            e=dpos_all[tk][p["entry_date"]]
            p["current"]=round(float(data[tk]["C"][i]),2)
            p["ret_pct"]=round((p["current"]-p["entry_price"])/p["entry_price"]*100,2)
            p["days_held"]=i-e+1
        p.pop("good",None)
    # 브레이커 상태 저장
    brk_pause=pause
    # 요약
    rets=[c["ret_pct"] for c in closed]
    a_cl=[c for c in closed if c["track"]=="A"]; b_cl=[c for c in closed if c["track"]=="B"]
    print(f"\n===== v14.8 백필 ({sim_days[0]}~{last}) =====")
    if rets:
        print(f"청산 {len(closed)}건 (A {len(a_cl)} / B {len(b_cl)}) | 승률 {sum(1 for r in rets if r>0)/len(rets)*100:.0f}% | 합산 {np.sum(rets):+.1f}%p")
        if a_cl: print(f"  A: 평균 {np.mean([c['ret_pct'] for c in a_cl]):+.2f}% 승률 {sum(1 for c in a_cl if c['ret_pct']>0)/len(a_cl)*100:.0f}%")
        if b_cl: print(f"  B: 평균 {np.mean([c['ret_pct'] for c in b_cl]):+.2f}% 승률 {sum(1 for c in b_cl if c['ret_pct']>0)/len(b_cl)*100:.0f}%")
    print(f"보유 {len(positions)}개 (A {sum(1 for p in positions if p['track']=='A')} / B {sum(1 for p in positions if p['track']=='B')}) | 브레이커 잔여 {brk_pause}일")
    for p in positions:
        print(f"  {p['track']} {p['name']} {p['entry_date']} {p['entry_price']:,.0f} → {p.get('current',0):,.0f} ({p.get('ret_pct',0):+.1f}%, {p.get('days_held')}일)")
    # history.json 교체 (legacy 보존)
    legacy=[]
    if os.path.exists("history.json"):
        old=json.load(open("history.json",encoding="utf-8"))
        legacy=old.get("legacy",[]) if isinstance(old,dict) else old
    hist={"format":"v14","backfilled":f"v14.8 {sim_days[0]}~{last}",
          "positions":positions,"closed":closed,"legacy":legacy,
          "breaker":{"pause_left":brk_pause,"reset_date":last if brk_pause>0 else ""}}
    json.dump(hist,open("history.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("\nhistory.json 갱신 완료")
    print("BF148_DONE")
