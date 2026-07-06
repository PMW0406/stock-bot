# -*- coding: utf-8 -*-
"""
주식봇 대시보드 — v14 52주 신고가 스윙
추천주(봇 스캔결과) + 히스토리(가상 포트폴리오) + 종목 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="주식봇 v14", page_icon="📈", layout="wide")

# ── v14 전략 상수 (morning_bot_cloud.py 와 동일) ──
SLOTS             = 12
HOLD_DAYS         = 15
STOP_LOSS         = 0.08
NEAR_HIGH         = -5.0
GAP_MAX           = 2.0
MARKET_CAP_MIN    = 100_000_000_000
MARKET_CAP_MAX    = 5_000_000_000_000
MIN_TRADING_VALUE = 3_000_000_000
REGIME_MA         = 120

BASE = os.path.dirname(__file__)


@st.cache_data
def load_stock_list():
    path = os.path.join(BASE, "stock_list.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def normalize_df(df):
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "open": rename[c] = "Open"
        elif cl == "high": rename[c] = "High"
        elif cl == "low": rename[c] = "Low"
        elif cl == "close": rename[c] = "Close"
        elif cl == "volume": rename[c] = "Volume"
        elif cl in ("adj close", "adj_close"): rename[c] = "Close"
    return df.rename(columns=rename)


def check_regime():
    try:
        df = fdr.DataReader("KS11", (datetime.today() - timedelta(days=300)).strftime("%Y-%m-%d"))
        close = df["Close"]
        c  = float(close.iloc[-1])
        ma = float(close.rolling(REGIME_MA).mean().iloc[-1])
        return c > ma, c, ma
    except:
        return None, None, None


def run_live_scan(progress):
    """실시간 v14 스캔 — 당일 거래대금 프리필터로 3~6분"""
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    kospi  = fdr.StockListing("KOSPI")
    kosdaq = fdr.StockListing("KOSDAQ")
    all_s  = pd.concat([kospi, kosdaq], ignore_index=True)
    filt   = all_s[(all_s["Marcap"] >= MARKET_CAP_MIN) & (all_s["Marcap"] <= MARKET_CAP_MAX)]
    # 프리필터: 당일 거래대금 10억↑ (20일평균 30억 후보의 안전 하한) → 대상 절반 이하로
    if "Amount" in filt.columns:
        filt = filt[filt["Amount"] >= 1_000_000_000]
    tickers = filt[["Code", "Name"]].values.tolist()
    total = len(tickers)
    results = []
    for i, (tk, nm) in enumerate(tickers):
        if i % 20 == 0:
            progress.progress(min(i / total, 1.0), text=f"스캔 중... {i}/{total} (후보 {len(results)}개)")
        try:
            df = fdr.DataReader(tk, start)
            if df.empty or len(df) < 260:
                continue
            close = df["Close"]; high = df["High"]; vol = df["Volume"]
            c   = float(close.iloc[-1])
            h52 = float(high.rolling(252).max().iloc[-1])
            if h52 <= 0: continue
            d52 = (c / h52 - 1) * 100
            if d52 < NEAR_HIGH: continue
            avg_value = float((vol * close).rolling(20).mean().iloc[-1])
            if np.isnan(avg_value) or avg_value < MIN_TRADING_VALUE: continue
            ma5  = float(close.rolling(5).mean().iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            if not (ma5 > ma20): continue
            results.append({
                "code": tk, "name": nm, "close": c,
                "d52": round(d52, 2), "avg_value_억": round(avg_value / 100_000_000, 1),
            })
        except:
            continue
    progress.progress(1.0, text="완료!")
    results.sort(key=lambda x: -x["d52"])
    return results


def analyze_stock_v14(ticker, name=""):
    """단일 종목 v14 조건 체크"""
    start = (datetime.today() - timedelta(days=420)).strftime("%Y-%m-%d")
    df = fdr.DataReader(ticker, start)
    if df is None or df.empty: return None
    df = normalize_df(df)
    if "Close" not in df.columns or len(df) < 260: return None

    close = df["Close"]; high = df["High"]; vol = df["Volume"]
    c    = float(close.iloc[-1])
    h52  = float(high.rolling(252).max().iloc[-1])
    d52  = (c / h52 - 1) * 100 if h52 > 0 else -99
    avg_value = float((vol * close).rolling(20).mean().iloc[-1])
    ma5  = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])

    conds = {
        f"① 52주 신고가 -{abs(NEAR_HIGH):.0f}% 이내": d52 >= NEAR_HIGH,
        "② 20일평균 거래대금 30억 이상": avg_value >= MIN_TRADING_VALUE,
        "③ MA5 > MA20 (단기추세)": ma5 > ma20,
    }
    return {
        "종목명": name or ticker, "현재가": c,
        "통과": all(conds.values()), "조건": conds,
        "세부": {
            "52주 최고가": f"{h52:,.0f}원",
            "신고가 대비": f"{d52:+.2f}%",
            "거래대금(20일)": f"{avg_value/100_000_000:,.0f}억",
            "MA5": f"{ma5:,.0f}", "MA20": f"{ma20:,.0f}",
        },
    }


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📈 주식봇 v14 — 52주 신고가 스윙")
st.caption(f"코스피>120일선 국면 + 신고가 -5%이내 + 거래대금30억 + MA5>MA20 | {HOLD_DAYS}일 보유·-8%손절·{SLOTS}슬롯 | 5년 백테스트 전 연도 플러스")

tab1, tab2, tab3 = st.tabs(["🏆 오늘의 후보", "💼 포트폴리오 & 히스토리", "🔍 종목 분석"])

# ── 탭1: 오늘의 후보 ─────────────────────
with tab1:
    cpath = os.path.join(BASE, "candidates.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("format") == "v14":
            st.caption(f"마지막 봇 실행: {saved.get('updated','?')}")
            if saved.get("regime_on"):
                st.success(f"🟢 {saved.get('regime_msg','')}")
            else:
                st.warning(f"🟡 {saved.get('regime_msg','')} — 약세장 현금 대기")
            cands = saved.get("candidates", [])
            newset = set(saved.get("new_entries", []))
            if cands:
                rows = [{
                    "매수": "🎯" if c["code"] in newset else "",
                    "종목명": c["name"], "코드": c["code"],
                    "현재가": f"{c['close']:,.0f}원",
                    "신고가 대비": f"{c['d52']:+.1f}%",
                    "거래대금": f"{c['avg_value_억']:,.0f}억",
                } for c in cands]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("🎯 = 봇이 오늘 매수한 종목 (가상 포트폴리오 기준) · 시가 갭 +2% 이상이면 매수 보류")
            else:
                st.info("조건 통과 후보 없음")
        else:
            st.info("이전 버전(v13) 결과 파일입니다. 내일 아침 봇 실행 후 v14 형식으로 갱신됩니다.")
    else:
        st.info("아직 스캔 결과가 없어요. 매일 아침 7:10 자동 업데이트됩니다.")

    st.divider()
    col_a, col_b = st.columns(2)
    # 수동 국면 체크 (가벼움)
    with col_a:
        if st.button("📡 지금 국면 확인 (몇 초)"):
            on, c, ma = check_regime()
            if on is None:
                st.error("코스피 데이터 조회 실패")
            elif on:
                st.success(f"🟢 코스피 {c:,.0f} > 120일선 {ma:,.0f} — 매매 가능 국면")
            else:
                st.warning(f"🟡 코스피 {c:,.0f} < 120일선 {ma:,.0f} — 현금 대기 국면")
    # 실시간 전체 스캔
    with col_b:
        live = st.button("🔄 지금 실시간 스캔 (약 3~6분)", type="primary")
    if live:
        on, kc, kma = check_regime()
        if on is False:
            st.warning(f"🟡 코스피 {kc:,.0f} < 120일선 {kma:,.0f} — 약세 국면이라 매수 대상 아님 (참고용으로 스캔은 진행)")
        prog = st.progress(0, text="종목 목록 수집 중...")
        results = run_live_scan(prog)
        prog.empty()
        st.session_state["live_scan"] = {
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "regime_on": bool(on), "results": results,
        }
    ls = st.session_state.get("live_scan")
    if ls:
        st.markdown(f"**⚡ 실시간 스캔 결과** ({ls['when']} 기준, {'🟢 매매국면' if ls['regime_on'] else '🟡 약세국면·참고용'})")
        if ls["results"]:
            rows = [{
                "종목명": c["name"], "코드": c["code"],
                "현재가": f"{c['close']:,.0f}원",
                "신고가 대비": f"{c['d52']:+.1f}%",
                "거래대금(20일)": f"{c['avg_value_억']:,.0f}억",
            } for c in ls["results"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("신고가 최근접순 정렬 · 매수는 다음날 시가 기준, 갭 +2% 이상이면 보류")
        else:
            st.info("현재 조건 통과 종목 없음 (신고가 -5% 이내 + MA5>MA20 종목이 시장에 없음)")


# ── 탭2: 포트폴리오 & 히스토리 ───────────
with tab2:
    hpath = os.path.join(BASE, "history.json")
    if not os.path.exists(hpath):
        st.info("아직 기록이 없어요. 봇이 매일 아침 자동으로 쌓아요.")
    else:
        with open(hpath, encoding="utf-8") as f:
            hist = json.load(f)

        if isinstance(hist, list):
            st.warning("이전 버전(v13) 히스토리입니다. 내일 아침 봇 실행 시 v14 형식으로 자동 전환됩니다.")
            hist = {"positions": [], "closed": [], "legacy": hist}

        positions = hist.get("positions", [])
        closed    = hist.get("closed", [])

        # ── 공통 스타일 (v13 카드 디자인 계승) ──
        st.markdown("""
<style>
.stock-card {
    background: #1a1a2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.stock-name { font-size: 16px; font-weight: 700; color: #e2e8f0; }
.stock-code { font-size: 12px; color: #718096; margin-left: 8px; }
.tag { display: inline-block; border-radius: 6px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
.mini-box { background:#0d1117; border-radius:8px; padding:10px 14px; }
.mini-title { color:#718096; font-size:11px; margin-bottom:4px; }
</style>""", unsafe_allow_html=True)

        def pct_color(x):
            if x is None: return "#888"
            return "#00c853" if x >= 0 else "#ff1744"

        # ── 요약 카드 ──
        if closed:
            rets = [c["ret_pct"] for c in closed if c.get("ret_pct") is not None]
            wins = [x for x in rets if x > 0]
            unreal = [p.get("ret_pct") for p in positions if p.get("ret_pct") is not None]
            cols = st.columns(4)
            metrics = [
                ("완료 거래", f"{len(closed)}건", f"만기 {sum(1 for c in closed if '만기' in c['reason'])} · 손절 {sum(1 for c in closed if '손절' in c['reason'])}", "#63b3ed"),
                ("승률", f"{len(wins)/len(rets)*100:.0f}%" if rets else "-", f"수익 {len(wins)} / 전체 {len(rets)}", "#63b3ed"),
                ("평균 수익률", f"{np.mean(rets):+.2f}%" if rets else "-", "거래당 실현 기준", pct_color(np.mean(rets) if rets else None)),
                ("보유중 평가", f"{np.mean(unreal):+.1f}%" if unreal else "-", f"{len(positions)}종목 평균 수익률", pct_color(np.mean(unreal) if unreal else None)),
            ]
            for col, (t, v, s, vc) in zip(cols, metrics):
                col.markdown(f"""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border:1px solid #0f3460;border-radius:16px;padding:20px 16px;text-align:center;">
                  <div style="color:#a0aec0;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:6px;">{t}</div>
                  <div style="font-size:26px;font-weight:700;color:{vc};">{v}</div>
                  <div style="color:#4a5568;font-size:11px;margin-top:4px;">{s}</div></div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ── 보유 포지션 카드 ──
        st.markdown(f"""
<div style="display:flex;align-items:center;margin:6px 0 12px 0;">
  <div style="background:#0f3460;border-radius:8px;padding:4px 14px;font-size:14px;font-weight:700;color:#63b3ed;">💼 보유 중</div>
  <div style="color:#4a5568;font-size:13px;margin-left:10px;">{len(positions)} / {SLOTS} 슬롯</div>
</div>""", unsafe_allow_html=True)
        if positions:
            for p in sorted(positions, key=lambda x: x.get("days_held", 0), reverse=True):
                pending = p.get("entry_price") is None
                ret  = p.get("ret_pct")
                cur  = p.get("current")
                held = p.get("days_held", 0) or 0
                barw = min(int(held / HOLD_DAYS * 100), 100)
                stop_txt = f"{p['stop_price']:,.0f}원" if p.get("stop_price") else "-"
                st.markdown(f"""
<div class="stock-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
      <span class="stock-name">{p['name']}</span>
      <span class="stock-code">{p['code']}</span>
      {'<span class="tag" style="background:#4a3800;color:#ffc107;margin-left:8px;">체결대기</span>' if pending else ''}
    </div>
    <div style="text-align:right;">
      <div style="color:#718096;font-size:11px;">현재가</div>
      <div style="font-size:18px;font-weight:700;color:{pct_color(ret)};">{f"{cur:,.0f}원" if cur else "-"} <span style="font-size:13px;">({f"{ret:+.1f}%" if ret is not None else "-"})</span></div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
    <div class="mini-box"><div class="mini-title">진입 ({p['entry_date'][5:]})</div>
      <div style="color:#e2e8f0;font-size:14px;font-weight:600;">{'대기' if pending else f"{p['entry_price']:,.0f}원"}</div></div>
    <div class="mini-box"><div class="mini-title">보유일 {held}/{HOLD_DAYS}</div>
      <div style="background:#2d3748;border-radius:4px;height:8px;margin-top:6px;"><div style="background:#63b3ed;width:{barw}%;height:8px;border-radius:4px;"></div></div></div>
    <div class="mini-box"><div class="mini-title">손절가 (-8%)</div>
      <div style="color:#ff1744;font-size:14px;font-weight:600;">{stop_txt}</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("보유 종목 없음 (약세장 대기 또는 시작 전)")

        # ── 청산 기록: 날짜별 카드 ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
<div style="display:flex;align-items:center;margin:6px 0 12px 0;">
  <div style="background:#0f3460;border-radius:8px;padding:4px 14px;font-size:14px;font-weight:700;color:#63b3ed;">📤 청산 기록</div>
</div>""", unsafe_allow_html=True)
        if closed:
            from itertools import groupby
            closed_sorted = sorted(closed, key=lambda x: x["exit_date"], reverse=True)
            show_all = st.toggle("전체 보기", value=False, help="끄면 최근 15건만 표시")
            items = closed_sorted if show_all else closed_sorted[:15]
            for exit_date, group in groupby(items, key=lambda x: x["exit_date"]):
                group = list(group)
                day_sum = sum(c["ret_pct"] for c in group)
                st.markdown(f"""
<div style="display:flex;align-items:center;margin:18px 0 8px 0;">
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:3px 12px;font-size:13px;font-weight:700;color:#63b3ed;">📅 {exit_date}</div>
  <div style="color:#4a5568;font-size:12px;margin-left:10px;">{len(group)}건 청산 · 합산 <span style="color:{pct_color(day_sum)};">{day_sum:+.1f}%</span></div>
</div>""", unsafe_allow_html=True)
                for c in group:
                    is_stop = "손절" in c["reason"]
                    tag_bg, tag_fg = ("#3d1a1a", "#ff6b6b") if is_stop else ("#1a3d2a", "#4ade80")
                    st.markdown(f"""
<div class="stock-card" style="padding:12px 18px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span class="stock-name" style="font-size:15px;">{c['name']}</span>
      <span class="stock-code">{c['code']}</span>
      <span class="tag" style="background:{tag_bg};color:{tag_fg};margin-left:8px;">{c['reason']}</span>
    </div>
    <div style="font-size:18px;font-weight:700;color:{pct_color(c['ret_pct'])};">{c['ret_pct']:+.2f}%</div>
  </div>
  <div style="color:#718096;font-size:12px;margin-top:6px;">
    {c['entry_date'][5:]} 진입 {c['entry_price']:,.0f}원 → {c['exit_date'][5:]} 청산 {c['exit_price']:,.0f}원
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("아직 청산된 거래가 없어요.")

        # ── 구버전 기록 ──
        legacy = hist.get("legacy", [])
        if legacy:
            with st.expander(f"📦 이전 전략(v13) 추천 기록 {sum(len(e.get('candidates',[])) for e in legacy)}건 보기"):
                for e in sorted(legacy, key=lambda x: x.get("date",""), reverse=True):
                    names = ", ".join(c["종목명"] for c in e.get("candidates", []))
                    st.markdown(f"- **{e.get('date')}**: {names}")


# ── 탭3: 종목 분석 ───────────────────────
with tab3:
    st.subheader("종목 v14 조건 체크")
    stock_list = load_stock_list()
    options = [f"{s['name']} ({s['code']})" for s in stock_list] if stock_list else []

    sel = st.selectbox("종목 검색", options, index=None, placeholder="종목명 입력...") if options else None
    manual = st.text_input("또는 종목코드 직접 입력", placeholder="예: 005930")

    ticker, name = None, ""
    if sel:
        name, code = sel.rsplit(" (", 1)
        ticker = code.rstrip(")")
    elif manual.strip():
        ticker = manual.strip()

    if ticker and st.button("분석하기", type="primary"):
        with st.spinner("분석 중..."):
            try:
                res = analyze_stock_v14(ticker, name)
                if res is None:
                    st.error("데이터 부족 또는 조회 실패 (상장 1년 미만 종목은 분석 불가)")
                else:
                    on, kc, kma = check_regime()
                    st.markdown(f"### {res['종목명']} — 현재가 {res['현재가']:,.0f}원")
                    if res["통과"] and on:
                        st.success("✅ v14 매수 조건 전부 통과 + 시장 국면 OK")
                    elif res["통과"]:
                        st.warning("종목 조건은 통과했으나 🟡 시장이 약세 국면 (코스피<120일선) — 매수 대기")
                    else:
                        st.error("❌ 조건 미통과")
                    for k, v in res["조건"].items():
                        st.markdown(f"- {'✅' if v else '❌'} {k}")
                    st.markdown("**세부 지표**")
                    st.table(pd.DataFrame([res["세부"]]).T.rename(columns={0: "값"}))
            except Exception as e:
                import traceback
                st.error(f"오류: {e}")
                st.code(traceback.format_exc())

    st.divider()
    with st.expander("📖 v14 전략 전문 보기"):
        st.markdown(f"""
**국면 게이트** — 코스피 종가 > 120일선일 때만 매매, 아니면 전량 현금

**매수 조건 (전부 충족)**
1. 시가총액 1,000억 ~ 5조
2. 20일 평균 거래대금 30억 이상
3. **52주 최고가 대비 -5% 이내** (핵심 팩터)
4. MA5 > MA20 (단기추세 보강)
5. 진입일 시가가 전일종가 +2% 이상 갭업이면 매수 취소

**청산 (조기 익절 없음 — 승자 태우기)**
- {HOLD_DAYS}거래일 만기 매도 or -8% 손절

**자금 운용** — 최대 {SLOTS}종목 균등 분산 (종목당 자금의 {100//SLOTS}%)

**검증 근거 (5년 백테스트, 2021.7~2026.7, 수수료 0.3% 반영)**
- 30개 기술 팩터 전수조사 중 유일하게 5년 전 연도에서 무작위 대비 우위를 유지한 팩터
- 학계 검증된 '52주 신고가 모멘텀' 이상현상과 일치 (George & Hwang 2004)
- 완전분산 기준 CAGR +10~18%, MDD -19~-31%, **전 연도 플러스** (2022 +9%, 2024 +12%)
- 현실 기대치: 12슬롯 기준 **CAGR 9~15%** (생존편향·미래 불확실성 감안해 보수적으로)

⚠️ 백테스트는 미래를 보장하지 않습니다. 실계좌 전 모의운용 권장.
        """)
