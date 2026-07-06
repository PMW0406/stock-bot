# -*- coding: utf-8 -*-
"""
전략 실험실: 남은 전략군 일괄 검증 (5년, 연도별, 수수료 0.3%)
 S1 낙주매매 / S2 변동성수축 돌파 / S3 거래대금 폭증 / S4 월말월초 캘린더
 잣대: 연도별 평균수익이 '같은 보유기간 무작위 진입(베이스라인)'을 일관되게 이겨야 채택.
"""
import pickle, numpy as np, sys
COST=0.3  # %
D=pickle.load(open("lab_cache.pkl","rb"))
tdays=D["tdays"]; data=D["data"]; dpm={d:i for i,d in enumerate(tdays)}
years=sorted(set(d[:4] for d in tdays))
print(f"패널: {len(data)}종목 × {len(tdays)}일 ({tdays[0]}~{tdays[-1]})\n")

# ── 공통: 종목별 지표 사전계산 ──
P={}
for tk,v in data.items():
    C=v["C"].astype(np.float64); O=v["O"].astype(np.float64)
    H=v["H"].astype(np.float64); L=v["L"].astype(np.float64); V=v["V"]
    n=len(C)
    if n<300: continue
    ret1=np.full(n,np.nan); ret1[1:]=C[1:]/C[:-1]-1
    val=C*V
    aval20=np.full(n,np.nan)
    cs=np.cumsum(val)
    aval20[20:]=(cs[20:]-cs[:-20])/20
    ma200=np.full(n,np.nan)
    cs2=np.cumsum(C)
    ma200[200:]=(cs2[200:]-cs2[:-200])/200
    hi20=np.full(n,np.nan); lo10=np.full(n,np.nan); hh10=np.full(n,np.nan)
    for i in range(20,n):
        hi20[i]=H[i-20:i].max()
    for i in range(10,n):
        lo10[i]=L[i-10:i].min(); hh10[i]=H[i-10:i].max()
    clloc=np.where(H-L>0,(C-L)/np.maximum(H-L,1e-9),0.5)
    P[tk]=dict(C=C,O=O,H=H,L=L,val=val,aval20=aval20,ma200=ma200,
               hi20=hi20,rng10=(hh10-lo10)/np.maximum(C,1e-9),ret1=ret1,clloc=clloc,
               dates=v["dates"],n=n)

def hold_ret(p,j,hold,stop=None):
    """신호일 j → 다음날 시가 진입 → hold일 후 종가 (종가손절 옵션)"""
    n=p["n"]
    if j+1>=n: return None
    buy=p["O"][j+1]
    if not np.isfinite(buy) or buy<=0: return None
    end=min(j+1+hold,n-1)
    if stop is not None:
        sp=buy*(1-stop)
        for k in range(j+1,end+1):
            if p["C"][k]<=sp: return p["C"][k]/buy-1
    return p["C"][end]/buy-1

# ── 베이스라인: 무작위 진입 같은 보유기간 (샘플) ──
rng=np.random.default_rng(42)
base={}
for hold in (1,3,5,10):
    rs=[]
    for tk,p in P.items():
        js=rng.integers(210,p["n"]-hold-2,size=8)
        for j in js:
            r=hold_ret(p,int(j),hold)
            if r is not None: rs.append((p["dates"][j][:4],r))
    base[hold]={}
    for y in years:
        arr=[x[1] for x in rs if x[0]==y]
        base[hold][y]=np.mean(arr)*100 if arr else np.nan

def report(tag, sigs, hold, stop=None):
    """sigs: list of (tk, j)"""
    rs=[]
    for tk,j in sigs:
        p=P[tk]; r=hold_ret(p,j,hold,stop)
        if r is not None: rs.append((p["dates"][j][:4], r*100-COST))
    if len(rs)<30:
        print(f"  {tag:40s}: 표본부족 ({len(rs)}건)"); return
    r=np.array([x[1] for x in rs]); yr=np.array([x[0] for x in rs])
    w=r>0; edges=[]
    for y in years:
        m=yr==y
        if m.sum()<10: edges.append(None); continue
        edges.append(r[m].mean()-base[hold].get(y,0))
    npos=sum(1 for e in edges if e is not None and e>0)
    nval=sum(1 for e in edges if e is not None)
    es=" ".join((f"{y[2:]}:{e:+5.2f}" if e is not None else f"{y[2:]}:  -  ") for y,e in zip(years,edges))
    print(f"  {tag:40s}: n{len(r):6d} 승률{w.mean()*100:4.0f}% 평균{r.mean():+5.2f}% | 엣지 {es} | {npos}/{nval}")

LIQ=3e9
# ── S1 낙주매매 ──
print("[S1] 낙주매매 (급락 다음날 반등) — 엣지=무작위 대비 %p")
for crash in (-0.12,-0.18):
    sigs=[(tk,j) for tk,p in P.items() for j in np.where((p["ret1"]<=crash)&(p["aval20"]>=LIQ))[0] if j>210]
    for hold in (1,3,5):
        report(f"하루 {int(crash*100)}%↓ → {hold}일 보유", sigs, hold)
sigs=[(tk,j) for tk,p in P.items() for j in np.where((p["ret1"]<=-0.12)&(p["aval20"]>=LIQ)&(p["C"]>p["ma200"]))[0] if j>210]
report("상승추세(200MA위) 중 -12%↓ → 3일", sigs, 3)

# ── S2 변동성 수축 돌파 (VCP) ──
print("\n[S2] 변동성 수축 후 20일 신고가 돌파")
for rng_max in (0.08,0.12):
    sigs=[]
    for tk,p in P.items():
        cond=(p["C"]>p["hi20"])&(p["rng10"]<rng_max)&(p["aval20"]>=LIQ)
        sigs+=[(tk,j) for j in np.where(cond)[0] if j>210]
    for hold,stop in ((10,0.08),(5,None)):
        report(f"수축<{int(rng_max*100)}% 돌파 → {hold}일"+(" -8%손절" if stop else ""), sigs, hold, stop)

# ── S3 거래대금 폭증 장대양봉 ──
print("\n[S3] 거래대금 폭증(5배) + 장대양봉(+5%↑, 고가마감)")
sigs=[]
for tk,p in P.items():
    cond=(p["val"]>5*p["aval20"])&(p["ret1"]>=0.05)&(p["clloc"]>=0.7)&(p["aval20"]>=LIQ)
    sigs+=[(tk,j) for j in np.where(cond)[0] if j>210]
for hold,stop in ((3,None),(10,0.08)):
    report(f"폭증 양봉 → {hold}일"+(" -8%손절" if stop else ""), sigs, hold, stop)
sigs2=[(tk,j) for tk,j in sigs if P[tk]["C"][j]>P[tk]["ma200"][j]]
report("+ 200MA 위 한정 → 10일 -8%손절", sigs2, 10, 0.08)

# ── S4 월말월초 (TOM) ──
print("\n[S4] 월말월초: 월 마지막날 종가신호 → 다음날 시가 → 3일 보유 (유동주 전체)")
month_of=[d[:7] for d in tdays]
last_of_month=set()
for i in range(len(tdays)-1):
    if month_of[i]!=month_of[i+1]: last_of_month.add(tdays[i])
sigs=[]
for tk,p in P.items():
    dset={d:i for i,d in enumerate(p["dates"])}
    for d in last_of_month:
        j=dset.get(d)
        if j and j>210 and p["aval20"][j]>=LIQ: sigs.append((tk,j))
report("월말 매수 → 3일", sigs, 3)
report("월말 매수 → 5일", sigs, 5)
print("\nLAB_TEST_DONE")
