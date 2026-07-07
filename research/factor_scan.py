# -*- coding: utf-8 -*-
"""
팩터 스캔: factor_cache.npz 로 각 팩터가 강세장에서 10일수익을 예측하는지,
 연도별로 '상위20% vs 무작위' 엣지를 봐서 5년 내내(2022·2024 포함) 이기는 게 있는지.
"""
import numpy as np
Z=np.load("factor_cache.npz", allow_pickle=True)
tdays=[d.decode() if isinstance(d,bytes) else str(d) for d in Z["tdays"]]
day=Z["day"]; ret10=Z["ret10"]
FEATS=["ret5","ret20","ret60","rsi2","rsi14","dist_hi20","dist_lo20","dist_hi52",
       "atrp","volr","c_ma20","c_ma60","ma20_60","ma5_20","down3"]
F={k:Z[k] for k in FEATS}
yr=np.array([tdays[d][:4] for d in day]); years=sorted(set(yr))
print(f"강세장 행 {len(day)} / 연도 {years}\n")

def edge_by_year(vals, high_is_signal=True):
    """각 연도: '팩터 상위20% 종목' 10일수익 - '그해 무작위 평균'. 방향 자동판정."""
    res={}
    for y in years:
        m=(yr==y)
        v=vals[m]; r=ret10[m]
        if len(v)<200: res[y]=None; continue
        thr=np.nanpercentile(v,80) if high_is_signal else np.nanpercentile(v,20)
        sel=(v>=thr) if high_is_signal else (v<=thr)
        if sel.sum()<20: res[y]=None; continue
        res[y]=r[sel].mean()-r.mean()
    return res

print(f"{'팩터(방향)':22s} | " + " ".join(f"{y[2:]:>6}" for y in years) + " | 판정")
rows=[]
for k in FEATS:
    for hi in (True,False):
        e=edge_by_year(F[k],hi)
        vals=[e[y] for y in years if e[y] is not None]
        if not vals: continue
        pos=sum(1 for x in vals if x>0)
        # 일관성: 유효연도 중 몇 개나 양수 + 평균
        avg=np.mean(vals)
        arrow="상위20%" if hi else "하위20%"
        cells=" ".join(f"{(e[y] if e[y] is not None else 0):+6.2f}" for y in years)
        verdict="★강건" if pos==len(vals) and len(vals)>=4 else ("○일부" if pos>=len(vals)-1 else "✗")
        rows.append((avg if pos>=len(vals)-1 else -99, f"  {k+'('+arrow+')':22s} | {cells} | {pos}/{len(vals)}양 {verdict}"))
# 강건한 것 위로 정렬
for _,line in sorted(rows,key=lambda x:-x[0]):
    print(line)
print("\n※ 모든 유효연도(특히 2022·2024)에 +면 '★강건' = 진짜 후보")
print("완료 / 총 done")
