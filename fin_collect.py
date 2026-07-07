# -*- coding: utf-8 -*-
"""
DART 분기 재무(영업이익·매출) 일괄 수집 — 다중회사 API (배치 100)
 대상: 유니버스(1000억~5조) + 초대형(5조↑) / 2021~2026 × 1Q·반기·3Q·사업
 저장: dart_fin.parquet (stock_code, bsns_year, reprt_code, account, thstrm, frmtrm)
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import os, time, pickle, requests, pandas as pd
from config import DART_API_KEY as KEY

mp=pickle.load(open("dart_corpmap.pkl","rb")); rev={v:k for k,v in mp.items()}
D=pickle.load(open("lab_cache.pkl","rb"))
mega=pickle.load(open("flows_price_cache.pkl","rb"))
stocks=sorted(set(list(D["data"].keys())+list(mega.keys())))
corps=[(rev[s],s) for s in stocks if s in rev]
print(f"대상 {len(corps)}사")
PARTIAL="fin_partial.pkl"; SAVE="dart_fin.parquet"
YEARS=["2021","2022","2023","2024","2025","2026"]
REPRT=["11013","11012","11014","11011"]   # 1Q 반기 3Q 사업
def num(x):
    try: return float(str(x).replace(",",""))
    except: return None
if os.path.exists(PARTIAL):
    st=pickle.load(open(PARTIAL,"rb")); done=st["done"]; rows=st["rows"]
    print(f"[재개] {len(done)}콜 완료")
else:
    done=set(); rows=[]
batches=[corps[i:i+100] for i in range(0,len(corps),100)]
t0=time.time()
for y in YEARS:
    for rc in REPRT:
        for bi,batch in enumerate(batches):
            key=(y,rc,bi)
            if key in done: continue
            codes=",".join(c for c,s in batch)
            smap={c:s for c,s in batch}
            try:
                r=requests.get("https://opendart.fss.or.kr/api/fnlttMultiAcnt.json",
                    params=dict(crtfc_key=KEY,corp_code=codes,bsns_year=y,reprt_code=rc),timeout=30)
                d=r.json()
            except Exception:
                time.sleep(2); continue
            if d.get("status")=="020":
                print("한도초과 60s 대기"); time.sleep(60); continue
            if d.get("status")=="000":
                for x in d["list"]:
                    if x.get("account_nm") not in ("영업이익","매출액"): continue
                    rows.append((smap.get(x["corp_code"],""), y, rc, x["account_nm"],
                                 x.get("fs_div",""), num(x.get("thstrm_amount")), num(x.get("frmtrm_amount"))))
            done.add(key)
            if len(done)%30==0:
                pickle.dump({"done":done,"rows":rows},open(PARTIAL,"wb"))
                print(f"  {len(done)}콜 (행 {len(rows):,}, {time.time()-t0:.0f}s)"); sys.stdout.flush()
            time.sleep(0.15)
df=pd.DataFrame(rows,columns=["stock_code","bsns_year","reprt_code","account","fs_div","thstrm","frmtrm"])
df.to_parquet(SAVE,index=False)
if os.path.exists(PARTIAL): os.remove(PARTIAL)
print(f"저장 {SAVE}: {len(df):,}행 / 종목 {df.stock_code.nunique()}")
print("FIN_DONE")
