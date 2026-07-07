# -*- coding: utf-8 -*-
"""
초대형주(시총 5조↑) 외국인/기관 수급 ~2.8년치 크롤링 (네이버, 체크포인트 재개)
 → flows_mega.parquet : ticker, date, frgn(주), inst(주)
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import os, time, types
class _P:
    def __getattr__(s,k): return _p
    def __setattr__(s,k,v): pass
    def __call__(s,*a,**k): return _p
_p=_P()
pl=types.ModuleType("plotly"); pl.__path__=[]; pl.__getattr__=lambda k:_p; sys.modules["plotly"]=pl
for s in ("io","graph_objects","express","subplots","offline","figure_factory"):
    m=types.ModuleType("plotly."+s); m.__getattr__=lambda k:_p; setattr(pl,s,m); sys.modules["plotly."+s]=m
import pandas as pd, requests
from io import StringIO
import FinanceDataReader as fdr
import warnings; warnings.filterwarnings("ignore")

PAGES=35            # 35페이지 ≈ 700거래일 ≈ 2.8년
SAVE="flows_mega.parquet"; TMP=SAVE+".tmp"
HEADERS={"User-Agent":"Mozilla/5.0"}

def fetch(ticker):
    rows=[]
    for page in range(1,PAGES+1):
        try:
            url=f"https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}"
            r=requests.get(url,headers=HEADERS,timeout=6)
            tables=pd.read_html(StringIO(r.text))
            got=False
            for t in tables:
                cols=str(t.columns.tolist())
                if "외국인" in cols and "기관" in cols:
                    t.columns=["_".join(c) if isinstance(c,tuple) else c for c in t.columns]
                    dc=next((c for c in t.columns if "날짜" in c),None)
                    ic=next((c for c in t.columns if "기관" in c and "순매" in c),None)
                    fc=next((c for c in t.columns if "외국인" in c and "순매" in c),None)
                    if dc:
                        t=t.dropna(subset=[dc])
                        for _,row in t.iterrows():
                            rows.append({"ticker":ticker,
                                "date":str(row[dc]).replace(".","-")[:10],
                                "inst":float(row[ic]) if ic and pd.notna(row[ic]) else 0,
                                "frgn":float(row[fc]) if fc and pd.notna(row[fc]) else 0})
                        got=True
                    break
            if not got: break
            time.sleep(0.05)
        except Exception:
            time.sleep(0.5)
    return rows

if __name__=="__main__":
    alls=pd.concat([fdr.StockListing("KOSPI"),fdr.StockListing("KOSDAQ")],ignore_index=True)
    mega=alls[alls["Marcap"]>=5_000_000_000_000][["Code","Name"]].values.tolist()
    print(f"초대형주 {len(mega)}개 × {PAGES}페이지 크롤링")
    if os.path.exists(TMP):
        prev=pd.read_parquet(TMP); done=set(prev["ticker"].unique()); all_rows=prev.to_dict("records")
        print(f"[재개] {len(done)}개 완료분 로드")
    else:
        done=set(); all_rows=[]
    t0=time.time()
    todo=[(c,n) for c,n in mega if c not in done]
    for i,(tk,nm) in enumerate(todo):
        rows=fetch(tk)
        all_rows+=rows
        if i%10==9 or i==len(todo)-1:
            pd.DataFrame(all_rows).to_parquet(TMP,index=False)
            print(f"  [체크포인트] {i+1}/{len(todo)} ({nm}, 누적 {len(all_rows):,}행, {time.time()-t0:.0f}s)"); sys.stdout.flush()
    df=pd.DataFrame(all_rows); df["date"]=pd.to_datetime(df["date"])
    df.to_parquet(SAVE,index=False)
    if os.path.exists(TMP): os.remove(TMP)
    print(f"저장: {SAVE} / {len(df):,}행 / 종목 {df['ticker'].nunique()} / {df['date'].min().date()}~{df['date'].max().date()}")
    print("FLOWS_DONE")
