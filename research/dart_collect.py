# -*- coding: utf-8 -*-
"""
DART 실적공시 이벤트 수집기 (PEAD 연구용)
 - corpCode 매핑(상장사만) + 2021-07~현재 월별 공시목록 스캔
 - 대상: 정기공시(분기/반기/사업보고서) + 거래소공시 중 잠정실적/손익구조변동
 - 저장: dart_events.parquet (stock_code, corp_name, report_nm, rcept_dt, rcept_no)
 - 체크포인트: dart_partial.pkl (월 단위 재개)
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import os, time, pickle, zipfile, re
import requests, pandas as pd
from io import BytesIO
from datetime import datetime
from config import DART_API_KEY as KEY

PARTIAL="dart_partial.pkl"; SAVE="dart_events.parquet"
KEYWORDS=("분기보고서","반기보고서","사업보고서","영업실적","잠정실적","매출액또는손익구조")

def corp_map():
    """corp_code → stock_code (상장사만)"""
    if os.path.exists("dart_corpmap.pkl"):
        return pickle.load(open("dart_corpmap.pkl","rb"))
    print("corpCode.xml 다운로드...")
    r=requests.get("https://opendart.fss.or.kr/api/corpCode.xml",
                   params=dict(crtfc_key=KEY),timeout=30)
    z=zipfile.ZipFile(BytesIO(r.content))
    xml=z.read(z.namelist()[0]).decode("utf-8")
    mp={}
    for m in re.finditer(r"<list>(.*?)</list>", xml, re.S):
        blk=m.group(1)
        cc=re.search(r"<corp_code>(.*?)</corp_code>",blk)
        sc=re.search(r"<stock_code>(.*?)</stock_code>",blk)
        if cc and sc and sc.group(1).strip():
            mp[cc.group(1).strip()]=sc.group(1).strip()
    pickle.dump(mp,open("dart_corpmap.pkl","wb"))
    print(f"  상장사 매핑 {len(mp):,}개")
    return mp

def month_ranges(start="2021-07-01"):
    out=[]; cur=pd.Timestamp(start)
    end=pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))
    while cur<=end:
        nxt=(cur+pd.offsets.MonthEnd(0))
        out.append((cur.strftime("%Y%m%d"), min(nxt,end).strftime("%Y%m%d")))
        cur=nxt+pd.Timedelta(days=1)
    return out

def fetch_month(bgn, end, ptype):
    rows=[]; page=1
    while True:
        try:
            r=requests.get("https://opendart.fss.or.kr/api/list.json",
                params=dict(crtfc_key=KEY,bgn_de=bgn,end_de=end,pblntf_ty=ptype,
                            page_no=page,page_count=100),timeout=15)
            d=r.json()
        except Exception:
            time.sleep(2); continue
        if d.get("status")=="020":     # 사용한도 초과
            print("  !! API 한도 초과 — 60초 대기"); time.sleep(60); continue
        if d.get("status")!="000": break
        for it in d.get("list",[]):
            nm=it.get("report_nm","")
            if any(k in nm for k in KEYWORDS):
                rows.append((it["corp_code"],it["corp_name"],nm,it["rcept_dt"],it["rcept_no"]))
        if page>=int(d.get("total_page",1)): break
        page+=1
        time.sleep(0.12)
    return rows

if __name__=="__main__":
    mp=corp_map()
    months=month_ranges()
    if os.path.exists(PARTIAL):
        st=pickle.load(open(PARTIAL,"rb")); done=st["done"]; all_rows=st["rows"]
        print(f"[재개] {len(done)}개월 완료분 로드 (누적 {len(all_rows):,}행)")
    else:
        done=set(); all_rows=[]
    t0=time.time()
    for bgn,end in months:
        key=bgn[:6]
        if key in done: continue
        n0=len(all_rows)
        for ptype in ("A","I"):     # 정기공시 / 거래소공시
            all_rows+=fetch_month(bgn,end,ptype)
        done.add(key)
        pickle.dump({"done":done,"rows":all_rows},open(PARTIAL,"wb"))
        print(f"  {key}: +{len(all_rows)-n0}건 (누적 {len(all_rows):,}, {time.time()-t0:.0f}s)"); sys.stdout.flush()
    # 상장사만 + 저장
    df=pd.DataFrame(all_rows,columns=["corp_code","corp_name","report_nm","rcept_dt","rcept_no"])
    df["stock_code"]=df["corp_code"].map(mp)
    df=df.dropna(subset=["stock_code"]).drop_duplicates(subset=["rcept_no"])
    df.to_parquet(SAVE,index=False)
    if os.path.exists(PARTIAL): os.remove(PARTIAL)
    print(f"\n저장: {SAVE} / {len(df):,}건 / 종목 {df['stock_code'].nunique()}개 / {df['rcept_dt'].min()}~{df['rcept_dt'].max()}")
    print("DART_DONE")
