"""
지난 1주일 추천주 히스토리 백필
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── 파라미터 ──────────────────────────────
MARKET_CAP_MIN    = 100_000_000_000
MARKET_CAP_MAX    = 5_000_000_000_000
MIN_TRADING_VALUE = 3_000_000_000
MIN_SCORE         = 70
STOP_LOSS         = 0.04

# ── 지난 7일 중 영업일 추출 ───────────────
def get_trading_days(n=22):
    days = []
    d = datetime.today() - timedelta(days=1)  # 어제부터
    while len(days) < n:
        if d.weekday() < 5:  # 월~금
            days.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    return sorted(days)

# ── v13.2 핵심 함수 ───────────────────────
def stock_weekly_ok_asof(df_slice):
    try:
        df2 = df_slice.copy()
        df2.index = pd.to_datetime(df2.index)
        wdf = df2.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
        if len(wdf) < 12: return False
        close = wdf["Close"]
        ma10  = close.rolling(10).mean()
        c, m10 = float(close.iloc[-1]), float(ma10.iloc[-1])
        if np.isnan(m10): return False
        ret13w = (float(df_slice["Close"].iloc[-1]) - float(df_slice["Close"].iloc[-65])) / float(df_slice["Close"].iloc[-65]) * 100 if len(df_slice) >= 65 else 0
        high52 = float(wdf["High"].iloc[-53:].max()) if len(wdf) >= 53 else float(wdf["High"].max())
        return c > m10 and ret13w > 0 and (c - high52) / high52 * 100 >= -40
    except:
        return False


def scan_day(target_date_str, all_tickers, stock_cache):
    """특정 날짜 기준으로 v13.2 스캔"""
    target_dt = pd.Timestamp(target_date_str)

    # 시장 필터
    kospi_ok = kosdaq_ok = False
    try:
        for code, attr in [("KS11", "kospi"), ("KQ11", "kosdaq")]:
            df_m = stock_cache.get(code)
            if df_m is None: continue
            df_slice = df_m[df_m.index <= target_dt]
            if len(df_slice) < 20: continue
            c   = float(df_slice["Close"].iloc[-1])
            m5  = float(df_slice["Close"].rolling(5).mean().iloc[-1])
            m20 = float(df_slice["Close"].rolling(20).mean().iloc[-1])
            ok  = c > m20 and m5 > m20
            if attr == "kospi": kospi_ok = ok
            else: kosdaq_ok = ok
    except:
        pass

    if not kospi_ok and not kosdaq_ok:
        print(f"  {target_date_str}: 시장 필터 미통과")
        return []

    all_ret20 = {}
    results   = []

    for ticker, name in all_tickers:
        try:
            df_full = stock_cache.get(ticker)
            if df_full is None: continue
            df = df_full[df_full.index <= target_dt]
            if len(df) < 65: continue

            close  = df["Close"]
            volume = df["Volume"]

            ma20_s = close.rolling(20).mean()
            ma60_s = close.rolling(60).mean()
            ma5_s  = close.rolling(5).mean()
            ma20_now = float(ma20_s.iloc[-1])
            ma60_now = float(ma60_s.iloc[-1])

            today_row = df.iloc[-1]
            c_now = float(today_row["Close"])

            # RS
            if len(df) > 21:
                ret20 = (c_now - float(close.iloc[-21])) / float(close.iloc[-21]) * 100
                all_ret20[ticker] = ret20

            # 스코어
            score = 0
            if c_now > ma20_now: score += 10
            if ma20_now > ma60_now: score += 10
            ma60_prev = float(ma60_s.iloc[-11]) if len(ma60_s) > 11 else np.nan
            if not np.isnan(ma60_prev) and ma60_now > ma60_prev: score += 10

            ret_20 = (c_now - float(close.iloc[-21])) / float(close.iloc[-21]) * 100 if len(close) > 21 else 0
            if ret_20 > 5: score += 20
            elif ret_20 > 0: score += 10

            recent_high = close.iloc[-8:-1].max()
            pullback    = (c_now - recent_high) / recent_high * 100
            if -8 <= pullback <= -3: score += 20
            elif -3 < pullback <= -1: score += 10

            vol_recent   = volume.iloc[-4:-1].mean()
            vol_before   = volume.iloc[-9:-4].mean()
            vol_decrease = vol_recent < vol_before * 0.9 if vol_before > 0 else False
            vol_5avg     = volume.iloc[-6:-1].mean()
            vol_ratio    = float(today_row["Volume"]) / vol_5avg if vol_5avg > 0 else 0
            if vol_decrease and vol_ratio >= 1.5: score += 20
            elif vol_ratio >= 1.5: score += 10

            prev_close = float(df.iloc[-2]["Close"])
            prev_ma5   = float(ma5_s.iloc[-2])
            if prev_close < prev_ma5 and c_now > float(ma5_s.iloc[-1]): score += 10

            w_ok = stock_weekly_ok_asof(df)
            if w_ok: score += 5

            h_now = float(today_row["High"]); l_now = float(today_row["Low"])
            cl = (c_now - l_now) / (h_now - l_now) if (h_now - l_now) > 0 else 0.5
            is_bearish_vol = (today_row["Close"] < today_row["Open"]) and (vol_ratio >= 2.0)
            avg_value = (volume.iloc[-21:-1] * close.iloc[-21:-1]).mean()

            must_pass = (
                c_now > ma20_now and
                ma20_now > ma60_now and
                -8 <= pullback <= -0.5 and
                1.0 <= vol_ratio <= 2.5 and
                vol_decrease and
                not is_bearish_vol and
                0.40 <= cl <= 0.85 and
                w_ok and
                avg_value >= MIN_TRADING_VALUE
            )

            if not must_pass or score < MIN_SCORE:
                continue

            results.append({
                "ticker": ticker,
                "name":   name,
                "score":  score,
                "price":  c_now,
                "ret20":  all_ret20.get(ticker, 0),
            })
        except:
            continue

    # RS 필터 상위 20%
    if all_ret20:
        thr = np.percentile(list(all_ret20.values()), 80)
        results = [r for r in results if r["ret20"] >= thr]

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    return results


# ── 메인 ──────────────────────────────────
if __name__ == "__main__":
    trading_days = get_trading_days(22)
    print(f"백필 대상: {trading_days}")

    data_start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    data_end   = datetime.today().strftime("%Y-%m-%d")

    print("\n종목 목록 수집 중...")
    kospi  = fdr.StockListing("KOSPI")
    kosdaq = fdr.StockListing("KOSDAQ")
    all_s  = pd.concat([kospi, kosdaq], ignore_index=True)
    filtered = all_s[
        (all_s["Marcap"] >= MARKET_CAP_MIN) &
        (all_s["Marcap"] <= MARKET_CAP_MAX)
    ]
    all_tickers = filtered[["Code","Name"]].values.tolist()
    print(f"  → {len(all_tickers)}개")

    # 데이터 로드 (1회)
    print("\n데이터 로드 중 (시간 좀 걸려요)...")
    stock_cache = {}
    load_targets = all_tickers + [("KS11","KOSPI"), ("KQ11","KOSDAQ")]
    for i, (ticker, _) in enumerate(load_targets):
        if i % 200 == 0:
            print(f"  {i}/{len(load_targets)}...")
        try:
            df = fdr.DataReader(ticker, data_start, data_end)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                stock_cache[ticker] = df
        except:
            continue
    print(f"  → {len(stock_cache)}개 로드 완료")

    # 날짜별 스캔
    history = []
    for d in trading_days:
        print(f"\n{d} 스캔 중...")
        results = scan_day(d, all_tickers, stock_cache)
        print(f"  → {len(results)}개 후보")
        if results:
            for r in results:
                print(f"     {r['name']} ({r['ticker']}) {r['price']:,.0f}원 {r['score']}점")
            history.append({
                "date": d,
                "candidates": [
                    {
                        "종목코드": r["ticker"],
                        "종목명":   r["name"],
                        "점수":     r["score"],
                        "추천가":   r["price"],
                        "목표가":   round(r["price"] * 1.10),
                        "손절가":   round(r["price"] * (1 - STOP_LOSS)),
                    }
                    for r in results
                ]
            })

    # 기존 history.json 병합
    hist_path = "history.json"
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            existing = json.load(f)
        existing_dates = {h["date"] for h in existing}
        for h in history:
            if h["date"] not in existing_dates:
                existing.append(h)
        history_final = sorted(existing, key=lambda x: x["date"])
    else:
        history_final = sorted(history, key=lambda x: x["date"])

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_final, f, ensure_ascii=False, indent=2)

    print(f"\n완료! history.json 저장 ({len(history_final)}일치)")
