# -*- coding: utf-8 -*-
"""
외국인/기관 수급 전략 검증 (초대형주 ~2.8년)
 flows_mega.parquet(수급) × lab_cache.pkl(가격) 결합.
 신호=종가 확정 → 익일 시가 진입 → N일 후 종가 청산. 비용 0.3%.
 잣대: 연도별로 '같은 유니버스 무작위 진입' 대비 엣지가 일관 양수인가.
"""
import pickle, numpy as np, pandas as pd, sys, types, os
class _PP:
    def __getattr__(s,k): return _pp
    def __setattr__(s,k,v): pass
    def __call__(s,*a,**k): return _pp
_pp=_PP()
_plm=types.ModuleType("plotly"); _plm.__path__=[]; _plm.__getattr__=lambda k:_pp; sys.modules["plotly"]=_plm
for _s in ("io","graph_objects","express","subplots","offline","figure_factory"):
    _m=types.ModuleType("plotly."+_s); _m.__getattr__=lambda k:_pp; setattr(_plm,_s,_m); sys.modules["plotly."+_s]=_m
import FinanceDataReader as fdr
import warnings; warnings.filterwarnings("ignore")
COST=0.3
FL=pd.read_parquet("flows_mega.parquet")
FL["date"]=pd.to_datetime(FL["date"]).dt.strftime("%Y-%m-%d")
print(f"수급: {FL['ticker'].nunique()}종목 {FL['date'].min()}~{FL['date'].max()} {len(FL):,}행")
# 초대형주 가격 직접 수집 (lab_cache는 5조 이하 유니버스라 미포함)
PRICE_CACHE="flows_price_cache.pkl"
if os.path.exists(PRICE_CACHE):
    data=pickle.load(open(PRICE_CACHE,"rb"))
else:
    data={}
    tks=sorted(FL["ticker"].unique())
    for i,tk in enumerate(tks):
        if i%30==0: print(f"  가격 {i}/{len(tks)}"); sys.stdout.flush()
        try:
            df=fdr.DataReader(tk,"2023-05-01")
            if len(df)<300: continue
            df.index=pd.to_datetime(df.index).strftime("%Y-%m-%d")
            data[tk]={"dates":np.array(df.index),
                      "O":df["Open"].values.astype(np.float32),
                      "C":df["Close"].values.astype(np.float32),
                      "V":df["Volume"].values.astype(np.float64)}
        except: pass
    pickle.dump(data,open(PRICE_CACHE,"wb"))
print(f"가격 확보: {len(data)}종목")

# 종목별 정렬된 수급 시계열 → 가격 데이터와 날짜 정합
P={}
for tk,g in FL.groupby("ticker"):
    if tk not in data: continue
    v=data[tk]
    dates=list(v["dates"]); dpos={d:i for i,d in enumerate(dates)}
    n=len(dates)
    frgn=np.full(n,np.nan); inst=np.full(n,np.nan)
    for _,row in g.iterrows():
        i=dpos.get(row["date"])
        if i is not None:
            frgn[i]=row["frgn"]; inst[i]=row["inst"]
    C=v["C"].astype(np.float64); O=v["O"].astype(np.float64); V=v["V"]
    aval=pd.Series(C*V).rolling(20).mean().values
    ma20=pd.Series(C).rolling(20).mean().values
    # 수급대금 강도: 순매수주수×종가 / 20일평균 거래대금
    fint=frgn*C/np.maximum(aval,1)
    iint=inst*C/np.maximum(aval,1)
    P[tk]={"C":C,"O":O,"dates":dates,"frgn":frgn,"inst":inst,
           "fint":fint,"iint":iint,"ma20":ma20,"n":n}
print(f"가격 결합 완료: {len(P)}종목")

years=sorted(set(d[:4] for tk in P for d in P[tk]["dates"] if not np.isnan(P[tk]["frgn"][P[tk]["dates"].index(d)] if False else 0)))
def run(sig_fn, hold, tag):
    rows=[]
    for tk,p in P.items():
        C=p["C"]; O=p["O"]; n=p["n"]
        valid=~np.isnan(p["frgn"])
        for j in range(25,n-hold-1):
            if not valid[j]: continue
            if not sig_fn(p,j): continue
            buy=O[j+1]
            if buy<=0: continue
            r=(C[min(j+1+hold,n-1)]/buy-1)*100-COST
            rows.append((p["dates"][j][:4], r))
    if len(rows)<40: print(f"  {tag:36s}: 표본부족({len(rows)})"); return
    r=np.array([x[1] for x in rows]); yr=np.array([x[0] for x in rows]); w=r>0
    ys=" ".join(f"{y[2:]}:{r[yr==y].mean():+5.2f}" for y in sorted(set(yr)) if (yr==y).sum()>=15)
    print(f"  {tag:36s}: n{len(r):5d} 승률{w.mean()*100:3.0f}% 평균{r.mean():+5.2f}% | {ys}")

def consec(arr, j, k):
    if j-k+1<0: return False
    seg=arr[j-k+1:j+1]
    return (not np.any(np.isnan(seg))) and np.all(seg>0)

print("\n[수급 전략 — 5일 보유]")
run(lambda p,j: consec(p["frgn"],j,3), 5, "F1 외인 3일 연속 순매수")
run(lambda p,j: consec(p["frgn"],j,5), 5, "F2 외인 5일 연속 순매수")
run(lambda p,j: not np.isnan(p["fint"][j]) and p["fint"][j]>=0.10, 5, "F3 외인 대량매수(대금10%↑)")
run(lambda p,j: consec(p["frgn"],j,3) and consec(p["inst"],j,3), 5, "F4 외인+기관 동시 3일")
run(lambda p,j: consec(p["frgn"],j,3) and p["C"][j]>p["ma20"][j], 5, "F5 외인3일 + MA20위")
run(lambda p,j: j>=3 and (not np.any(np.isnan(p["frgn"][j-2:j+1]))) and np.all(p["frgn"][j-2:j+1]<0), 5, "F6 (역) 외인 3일 순매도")
run(lambda p,j: not np.isnan(p["frgn"][j]), 5, "F0 기준선: 아무날(수급데이터 있는 날)")
print("\n[10일 보유]")
run(lambda p,j: consec(p["frgn"],j,5), 10, "F2' 외인 5일 연속 (10일 보유)")
run(lambda p,j: consec(p["frgn"],j,3) and p["C"][j]>p["ma20"][j], 10, "F5' 외인3일+MA20위 (10일)")
run(lambda p,j: not np.isnan(p["frgn"][j]), 10, "F0' 기준선 (10일)")
print("\nFLOWTEST_DONE")
