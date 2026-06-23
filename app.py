"""
주식봇 대시보드 - v13.2 전략
추천주 + 종목 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="주식봇 v13.2", page_icon="📈", layout="wide")

# 종목 리스트 로드 (자동완성용)
@st.cache_data
def load_stock_list():
    path = os.path.join(os.path.dirname(__file__), "stock_list.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []

MARKET_CAP_MIN    = 100_000_000_000
MARKET_CAP_MAX    = 5_000_000_000_000
MIN_TRADING_VALUE = 3_000_000_000
MIN_SCORE         = 70

# ─────────────────────────────────────────
# 공통 함수
# ─────────────────────────────────────────
def stock_weekly_ok(df):
    try:
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index)
        wdf = df2.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
        if len(wdf) < 12: return False
        close = wdf["Close"]
        ma10  = close.rolling(10).mean()
        c, m10 = float(close.iloc[-1]), float(ma10.iloc[-1])
        if np.isnan(m10): return False
        ret13w = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-65])) / float(df["Close"].iloc[-65]) * 100 if len(df) >= 65 else 0
        high52 = float(wdf["High"].iloc[-53:].max()) if len(wdf) >= 53 else float(wdf["High"].max())
        return c > m10 and ret13w > 0 and (c - high52) / high52 * 100 >= -40
    except:
        return False


def normalize_df(df):
    """컬럼명 대문자 통일"""
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":   rename[c] = "Open"
        elif cl == "high":   rename[c] = "High"
        elif cl == "low":    rename[c] = "Low"
        elif cl == "close":  rename[c] = "Close"
        elif cl == "volume": rename[c] = "Volume"
        elif cl in ("adj close", "adj_close"): rename[c] = "Close"
    return df.rename(columns=rename)


def analyze_stock(ticker, name=""):
    """단일 종목 v13.2 분석"""
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    df = fdr.DataReader(ticker, start, end)
    if df is None or df.empty:
        return None
    df = normalize_df(df)
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    if len(df) < 65:
        return None
    df.index = pd.to_datetime(df.index)

    today = df.iloc[-1]
    close = df["Close"]; volume = df["Volume"]
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma5_now  = float(ma5.iloc[-1])
    ma20_now = float(ma20.iloc[-1])
    ma60_now = float(ma60.iloc[-1])

    score = 0
    if today["Close"] > ma20_now: score += 10
    if ma20_now > ma60_now:       score += 10
    ma60_prev = float(ma60.iloc[-11]) if len(ma60) > 11 else np.nan
    if not np.isnan(ma60_prev) and ma60_now > ma60_prev: score += 10

    ret_20 = (today["Close"] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) > 21 else 0
    if ret_20 > 5:   score += 20
    elif ret_20 > 0: score += 10

    recent_high = close.iloc[-8:-1].max()
    pullback    = (today["Close"] - recent_high) / recent_high * 100
    if -8 <= pullback <= -3:   score += 20
    elif -3 < pullback <= -1: score += 10

    vol_recent   = volume.iloc[-4:-1].mean()
    vol_before   = volume.iloc[-9:-4].mean()
    vol_decrease = vol_recent < vol_before * 0.9 if vol_before > 0 else False
    vol_5avg     = volume.iloc[-6:-1].mean()
    vol_ratio    = float(today["Volume"]) / vol_5avg if vol_5avg > 0 else 0
    if vol_decrease and vol_ratio >= 1.5: score += 20
    elif vol_ratio >= 1.5:                score += 10

    prev_close = float(df.iloc[-2]["Close"])
    prev_ma5   = float(ma5.iloc[-2])
    ma5_recov  = prev_close < prev_ma5 and today["Close"] > ma5_now
    if ma5_recov: score += 10

    w_ok = stock_weekly_ok(df)
    if w_ok: score += 5

    h_now = float(df["High"].iloc[-1]); l_now = float(df["Low"].iloc[-1])
    cl = (today["Close"] - l_now) / (h_now - l_now) if (h_now - l_now) > 0 else 0.5

    is_bearish_vol = (today["Close"] < today["Open"]) and (vol_ratio >= 2.0)
    avg_value = (volume.iloc[-21:-1] * close.iloc[-21:-1]).mean()

    conds = {
        "① 주봉 MA10 위 + 13주↑ + 52주고점-40%이내": w_ok,
        "② 종가 > MA20 > MA60": today["Close"] > ma20_now and ma20_now > ma60_now,
        "③ 풀백 -0.5%~-8%": -8 <= pullback <= -0.5,
        "④ 거래량비 1.0~2.5배": 1.0 <= vol_ratio <= 2.5,
        "⑤ 거래량 감소 중": vol_decrease,
        "⑥ 음봉+거래량 폭발 없음": not is_bearish_vol,
        "⑦ 종가위치 0.40~0.85": 0.40 <= cl <= 0.85,
        "⑧ 거래대금 30억 이상": avg_value >= MIN_TRADING_VALUE,
    }
    must_pass = all(conds.values())

    return {
        "종목명": name or ticker,
        "현재가": float(today["Close"]),
        "점수": score,
        "must_pass": must_pass,
        "조건": conds,
        "세부": {
            "MA20": f"{ma20_now:,.0f}",
            "MA60": f"{ma60_now:,.0f}",
            "풀백": f"{pullback:.1f}%",
            "거래량비": f"{vol_ratio:.2f}배",
            "종가위치": f"{cl:.2f}",
            "20일수익": f"{ret_20:.1f}%",
            "거래대금(20일평균)": f"{avg_value/100_000_000:.1f}억",
            "주봉": "OK ✅" if w_ok else "미통과 ❌",
            "MA5회복": "✅" if ma5_recov else "-",
        }
    }


def run_full_scan():
    """전체 스캔 (20~30분 소요)"""
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    prog = st.progress(0, text="시장 상태 확인 중...")

    # 시장 필터
    kospi_ok = kosdaq_ok = False
    try:
        for code, attr in [("KS11", "kospi"), ("KQ11", "kosdaq")]:
            mdf = fdr.DataReader(code, (datetime.today()-timedelta(days=60)).strftime("%Y-%m-%d"), end)
            c   = float(mdf["Close"].iloc[-1])
            m5  = float(mdf["Close"].rolling(5).mean().iloc[-1])
            m20 = float(mdf["Close"].rolling(20).mean().iloc[-1])
            ok  = c > m20 and m5 > m20
            if attr == "kospi": kospi_ok = ok
            else: kosdaq_ok = ok
    except:
        pass

    if not kospi_ok and not kosdaq_ok:
        prog.empty()
        return None, "시장 필터 미통과 (KOSPI·KOSDAQ 모두 MA20 아래)"

    prog.progress(5, text="종목 목록 수집 중...")
    all_s = pd.concat([fdr.StockListing("KOSPI"), fdr.StockListing("KOSDAQ")], ignore_index=True)
    filtered = all_s[(all_s["Marcap"] >= MARKET_CAP_MIN) & (all_s["Marcap"] <= MARKET_CAP_MAX)]
    if not kospi_ok:
        filtered = filtered[filtered.get("Market", "") != "KOSPI"]
    if not kosdaq_ok:
        filtered = filtered[filtered.get("Market", "") != "KOSDAQ"]

    tickers  = filtered[["Code","Name"]].values.tolist()
    total    = len(tickers)
    all_ret20 = {}
    results   = []

    for i, (ticker, name) in enumerate(tickers):
        pct = int(5 + (i / total) * 90)
        if i % 50 == 0:
            prog.progress(pct, text=f"스캔 중... {i}/{total}")
        try:
            df = fdr.DataReader(ticker, start, end)
            if df.empty or len(df) < 65: continue
            df.index = pd.to_datetime(df.index)
            ret20 = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-21])) / float(df["Close"].iloc[-21]) * 100
            all_ret20[ticker] = ret20
            res = analyze_stock(ticker, name)
            if res and res["must_pass"] and res["점수"] >= MIN_SCORE:
                res["ret20"] = ret20
                results.append(res)
        except:
            continue

    if all_ret20:
        thr = np.percentile(list(all_ret20.values()), 80)
        results = [r for r in results if r.get("ret20", 0) >= thr]

    results = sorted(results, key=lambda x: x["점수"], reverse=True)[:10]
    prog.progress(100, text="완료!")
    prog.empty()
    return results, f"KOSPI {'✅' if kospi_ok else '❌'} / KOSDAQ {'✅' if kosdaq_ok else '❌'}"


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📈 주식봇 v13.2")
st.caption("추세+눌림목+거래량+주봉 전략 | 백테스트 승률 71.6% | 평균수익 +16.26%")

tab1, tab2, tab3 = st.tabs(["🏆 추천주", "📅 히스토리", "🔍 종목 분석"])

# ── 탭1: 추천주 ───────────────────────────
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("오늘의 스윙 후보")

    # 저장된 결과 불러오기
    if os.path.exists("candidates.json"):
        with open("candidates.json", encoding="utf-8") as f:
            saved = json.load(f)
        with col2:
            st.caption(f"마지막 업데이트: {saved.get('updated','?')}")

        market_info = saved.get("market", "")
        kospi_ok = saved.get("kospi_ok", False)
        kosdaq_ok = saved.get("kosdaq_ok", False)
        st.info(f"시장: {market_info} | KOSPI {'✅' if kospi_ok else '❌'} KOSDAQ {'✅' if kosdaq_ok else '❌'}")

        cands = saved.get("candidates", [])
        if cands:
            rows = []
            for r in cands:
                d = r.get("detail", {})
                rows.append({
                    "종목명": r["종목명"],
                    "점수": f"{r['점수']}점",
                    "현재가": f"{r['현재가']:,.0f}원",
                    "목표가": f"{r['목표가']:,.0f}원",
                    "손절가": f"{r['손절가']:,.0f}원",
                    "풀백": d.get("눌림폭", "-"),
                    "거래량": d.get("거래량", "-"),
                    "주봉": d.get("주봉", "-"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("오늘은 조건 통과 종목 없음")
    else:
        st.info("아직 스캔 결과가 없어요. 아래 버튼으로 지금 스캔하거나 내일 아침 7:10 자동 업데이트를 기다려주세요.")

    st.divider()
    st.caption("⚠️ 지금 스캔은 20~30분 걸려요. 매일 아침 7:10에 자동으로 업데이트됩니다.")
    if st.button("🔄 지금 스캔하기", type="primary"):
        with st.spinner("스캔 중... (20~30분 소요)"):
            results, market_msg = run_full_scan()
        if results is None:
            st.error(f"❌ {market_msg}")
        elif not results:
            st.warning(f"시장: {market_msg}\n\n조건 통과 종목 없음")
        else:
            st.success(f"✅ {market_msg} | {len(results)}개 후보 발견")
            rows = []
            for r in results:
                rows.append({
                    "종목명": r["종목명"],
                    "점수": f"{r['점수']}점",
                    "현재가": f"{r['현재가']:,.0f}원",
                    "풀백": r["세부"].get("풀백", "-"),
                    "거래량비": r["세부"].get("거래량비", "-"),
                    "20일수익": r["세부"].get("20일수익", "-"),
                    "주봉": r["세부"].get("주봉", "-"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── 수익률 판정 함수 ──────────────────────
def evaluate_trade(ticker, rec_date_str, 추천가, 목표가, 손절가):
    """
    추천일 다음날부터 10거래일간 체크
    → 목표가 먼저 터치: 목표달성
    → 손절가 먼저 터치: 손절
    → 10일 경과: 종료(당시 종가)
    → 아직 10일 미경과: 보유중(현재가)
    """
    try:
        start = (pd.Timestamp(rec_date_str) + timedelta(days=1)).strftime("%Y-%m-%d")
        end   = datetime.today().strftime("%Y-%m-%d")
        df = fdr.DataReader(ticker, start, end)
        if df is None or df.empty:
            return None, None, "데이터없음"
        df = normalize_df(df)
        df.index = pd.to_datetime(df.index)
        trading_days = df.index.tolist()

        for i, dt in enumerate(trading_days[:10]):
            row = df.loc[dt]
            high = float(row["High"])
            low  = float(row["Low"])
            # 당일 고가가 목표가 터치
            if high >= 목표가:
                pct = (목표가 - 추천가) / 추천가 * 100
                return 목표가, pct, "🎯 목표달성"
            # 당일 저가가 손절가 터치
            if low <= 손절가:
                pct = (손절가 - 추천가) / 추천가 * 100
                return 손절가, pct, "🛑 손절"

        # 10일 경과 여부
        if len(trading_days) >= 10:
            final_price = float(df.iloc[9]["Close"])
            pct = (final_price - 추천가) / 추천가 * 100
            return final_price, pct, "📋 종료"
        else:
            current = float(df.iloc[-1]["Close"])
            pct = (current - 추천가) / 추천가 * 100
            return current, pct, "🔵 보유중"
    except:
        return None, None, "-"


# ── 탭2: 히스토리 ────────────────────────
with tab2:
    st.subheader("추천주 히스토리 & 수익률")

    hist_path = os.path.join(os.path.dirname(__file__), "history.json")
    if not os.path.exists(hist_path):
        st.info("아직 히스토리가 없어요. 매일 아침 7:10 자동으로 쌓여요.")
    else:
        with open(hist_path, encoding="utf-8") as f:
            history = json.load(f)

        if not history:
            st.info("추천 기록이 없어요.")
        else:
            history_sorted = sorted(history, key=lambda x: x["date"], reverse=True)

            # 전체 요약 통계
            all_rows = []
            for entry in history_sorted:
                for c in entry.get("candidates", []):
                    all_rows.append((entry["date"], c))

            if all_rows:
                with st.spinner("수익률 계산 중..."):
                    summary = []
                    for date_str, c in all_rows:
                        price, pct, status = evaluate_trade(
                            c["종목코드"], date_str, c["추천가"], c["목표가"], c["손절가"]
                        )
                        summary.append((date_str, c, price, pct, status))

                completed = [(d,c,p,pct,s) for d,c,p,pct,s in summary if s in ("🎯 목표달성","🛑 손절","📋 종료")]
                if completed:
                    wins  = [x for x in completed if x[4] == "🎯 목표달성"]
                    total = len(completed)
                    win_rate = len(wins) / total * 100
                    avg_pct  = np.mean([x[3] for x in completed if x[3] is not None])
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("총 추천", f"{len(summary)}건")
                    c2.metric("완료", f"{total}건")
                    c3.metric("실전 승률", f"{win_rate:.0f}%")
                    c4.metric("평균 수익률", f"{avg_pct:+.1f}%")
                    st.divider()

            # 날짜별 상세
            summary_by_date = {}
            for date_str, c, price, pct, status in summary:
                summary_by_date.setdefault(date_str, []).append((c, price, pct, status))

            for entry in history_sorted:
                date_str = entry["date"]
                cands    = entry.get("candidates", [])
                if not cands: continue

                rows = []
                for c, price, pct, status in summary_by_date.get(date_str, []):
                    추천가 = c["추천가"]
                    rows.append({
                        "종목명":  c["종목명"],
                        "상태":    status,
                        "추천가":  f"{추천가:,.0f}원",
                        "현재/종료가": f"{price:,.0f}원" if price else "-",
                        "수익률":  f"{pct:+.1f}%" if pct is not None else "-",
                        "목표가":  f"{c['목표가']:,.0f}원",
                        "손절가":  f"{c['손절가']:,.0f}원",
                        "점수":    f"{c['점수']}점",
                    })

                label = f"📅 {date_str}  —  {len(cands)}개 추천"
                with st.expander(label, expanded=(date_str == history_sorted[0]["date"])):
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── 탭3: 종목 분석 ────────────────────────
with tab3:
    st.subheader("종목 v13.2 조건 체크")

    stock_list = load_stock_list()
    stock_options = {f"{s['name']} ({s['code']})": s['code'] for s in stock_list}

    search_query = st.text_input("종목명 검색", placeholder="예: SK, 삼성, 카카오...")

    selected_label = None
    if search_query:
        matches = [label for label in stock_options if search_query.lower() in label.lower()][:20]
        if matches:
            selected_label = st.selectbox("종목 선택", matches)
        else:
            st.warning("검색 결과 없음")

    run_btn = st.button("분석", type="primary", disabled=(selected_label is None))

    if run_btn and selected_label:
        ticker = stock_options[selected_label]
        name   = selected_label.split(" (")[0]
        with st.spinner("데이터 수집 중..."):
            try:
                result = analyze_stock(ticker, name)

                if result is None:
                    st.error("데이터를 불러올 수 없어요. (데이터 부족 또는 상장 기간 짧은 종목)")
                else:
                    # 헤더
                    verdict_color = "🟢" if result["must_pass"] else "🔴"
                    verdict_text  = "매수 후보 ✅" if result["must_pass"] else "조건 미충족 ❌"
                    st.markdown(f"### {verdict_color} {result['종목명']} ({ticker})")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("현재가", f"{result['현재가']:,.0f}원")
                    c2.metric("스코어", f"{result['점수']}점")
                    c3.metric("결론", verdict_text)

                    # 조건 체크표
                    st.divider()
                    st.markdown("**📋 조건 체크**")
                    for cond, ok in result["조건"].items():
                        icon = "✅" if ok else "❌"
                        st.markdown(f"{icon} {cond}")

                    # 세부 수치
                    st.divider()
                    st.markdown("**📊 세부 수치**")
                    detail_df = pd.DataFrame([result["세부"]]).T.reset_index()
                    detail_df.columns = ["항목", "값"]
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

                    # 스코어 설명
                    st.divider()
                    score = result["점수"]
                    if result["must_pass"] and score >= 90:
                        st.success(f"🔥 {score}점 — 강한 신호! 백테스트 기준 이 구간 승률 73.7%")
                    elif result["must_pass"] and score >= 70:
                        st.success(f"✅ {score}점 — 매수 후보. 승률 70~72% 구간")
                    else:
                        fails = [k for k, v in result["조건"].items() if not v]
                        st.error(f"❌ 조건 미충족 ({len(fails)}개 탈락)")
                        for f in fails:
                            st.markdown(f"  - {f}")

            except Exception as e:
                import traceback
                st.error(f"오류: {e}")
                st.code(traceback.format_exc())

    st.divider()
    with st.expander("📖 v13.2 전략 조건 보기"):
        st.markdown("""
**매수 필수 조건 (전부 충족해야 함)**
1. 주봉 종가 > 주봉 MA10 + 13주 수익률 > 0 + 52주 고점 -40% 이내
2. 종가 > MA20 > MA60
3. 최근 7일 고점 대비 -0.5% ~ -8% 풀백
4. 거래량비 1.0 ~ 2.5배 (5일 평균 대비)
5. 최근 거래량 감소 중
6. 음봉+거래량 폭발 없음
7. 종가위치 0.40 ~ 0.85
8. 20일 평균 거래대금 30억 이상

**스코어 (70점 이상 + RS 상위 20% = 최종 후보)**
- 트렌드 30점 / 상대강도 20점 / 눌림목 20점 / 거래량 20점 / MA5회복 10점 / 주봉보너스 5점

**매매 원칙**
- 손절: -4% / 보유: 최대 10거래일 / 비중: 자금의 20% / 최대 5종목
        """)
