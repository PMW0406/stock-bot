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
STOP_LOSS         = 0.10   # 종가 기준
NEAR_HIGH         = -5.0
NEAR_HIGH_TOP     = -1.0
PREM_MIN, PREM_MAX   = 4.0, 8.0
RET20_MIN, RET20_MAX = 10.0, 25.0
VOL_SPIKE_MAX     = 4.0
FRESH_DAYS        = 5
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
    """코스피 vs 120일선 + 5일 확인 히스테리시스 (봇과 동일)"""
    try:
        df = fdr.DataReader("KS11", (datetime.today() - timedelta(days=420)).strftime("%Y-%m-%d"))
        close = df["Close"]
        ma = close.rolling(REGIME_MA).mean()
        raw = (close > ma).dropna().tolist()
        state = raw[0]; cnt = 0
        for x in raw[1:]:
            if x != state:
                cnt += 1
                if cnt >= 5: state = x; cnt = 0
            else:
                cnt = 0
        return bool(state), float(close.iloc[-1]), float(ma.iloc[-1])
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
            if not (NEAR_HIGH <= d52 < NEAR_HIGH_TOP): continue
            avg_value = float((vol * close).rolling(20).mean().iloc[-1])
            if np.isnan(avg_value) or avg_value < MIN_TRADING_VALUE: continue
            ma5  = float(close.rolling(5).mean().iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            if np.isnan(ma20) or ma20 <= 0: continue
            prem = (ma5 / ma20 - 1) * 100
            if not (PREM_MIN <= prem < PREM_MAX): continue
            ret20 = (c - float(close.iloc[-21])) / float(close.iloc[-21]) * 100
            if not (RET20_MIN <= ret20 < RET20_MAX): continue
            vol5 = float(vol.iloc[-6:-1].mean())
            if vol5 > 0 and float(vol.iloc[-1]) / vol5 >= VOL_SPIKE_MAX: continue
            if float(high.iloc[-FRESH_DAYS:].max()) < h52 * 0.9999: continue
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

    prem  = (ma5 / ma20 - 1) * 100 if ma20 > 0 else 0
    ret20 = (c - float(close.iloc[-21])) / float(close.iloc[-21]) * 100
    vol5  = float(vol.iloc[-6:-1].mean())
    vr    = float(vol.iloc[-1]) / vol5 if vol5 > 0 else 0
    conds = {
        "① 52주 신고가 -5% ~ -1% 밴드": NEAR_HIGH <= d52 < NEAR_HIGH_TOP,
        "② 20일평균 거래대금 30억 이상": avg_value >= MIN_TRADING_VALUE,
        "③ MA5/MA20 이격 +4~8% (추세형성)": PREM_MIN <= prem < PREM_MAX,
        "④ 20일 수익률 +10~25% (모멘텀 밴드)": RET20_MIN <= ret20 < RET20_MAX,
        "⑤ 거래량 폭증(5일평균 4배↑) 아님": not (vr >= VOL_SPIKE_MAX),
        "⑥ 신고가가 최근 5일 내 갱신 (신선도)": float(high.iloc[-FRESH_DAYS:].max()) >= h52 * 0.9999,
    }
    return {
        "종목명": name or ticker, "현재가": c,
        "통과": all(conds.values()), "조건": conds,
        "세부": {
            "52주 최고가": f"{h52:,.0f}원",
            "신고가 대비": f"{d52:+.2f}%",
            "거래대금(20일)": f"{avg_value/100_000_000:,.0f}억",
            "MA5/MA20 이격": f"{prem:+.1f}%", "20일 수익률": f"{ret20:+.1f}%", "거래량 배율": f"{vr:.1f}배",
        },
    }


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📈 주식봇 v14 — 52주 신고가 스윙")
st.caption(f"v14.7: 신선한 신고가 + 밴드정밀화 + 실적성장 필터(역성장 제외·고성장 우선) | {HOLD_DAYS}일·종가-10%손절·{SLOTS}슬롯 + B트랙(초대형회귀) | 5년 전 연도·전 슬롯 플러스")

tab1, tab2, tab3 = st.tabs(["🏆 오늘의 후보", "💼 포트폴리오 & 히스토리", "🔍 종목 분석"])

# ── 탭1: 오늘의 후보 ─────────────────────
with tab1:
    cpath = os.path.join(BASE, "candidates.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("format") == "v14":
            st.markdown("""
<style>
.cand-card { background:#1a1a2e; border:1px solid #2d3748; border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.cand-card.buy { border:1px solid #00c853; background:linear-gradient(135deg,#16281c 0%,#1a1a2e 60%); }
.cand-card.buyb { border:1px solid #2d6cdf; background:linear-gradient(135deg,#14223a 0%,#1a1a2e 60%); }
.cand-name { font-size:16px; font-weight:700; color:#e2e8f0; }
.cand-code { font-size:11px; color:#718096; margin-left:6px; }
.pill { display:inline-block; border-radius:6px; padding:2px 9px; font-size:11px; font-weight:700; margin-left:6px; }
.metric { display:inline-block; background:#0d1117; border-radius:6px; padding:4px 10px; margin:6px 6px 0 0; font-size:12px; color:#9fb0c3; }
.metric b { color:#e2e8f0; }
</style>""", unsafe_allow_html=True)
            # 국면 배너
            rg = saved.get("regime_msg", "")
            if saved.get("regime_on"):
                st.markdown(f"""<div style="background:linear-gradient(135deg,#0d2818 0%,#16213e 100%);border:1px solid #00c853;border-radius:14px;padding:14px 20px;margin-bottom:14px;">
                  <span style="font-size:15px;font-weight:700;color:#4ade80;">🟢 {rg}</span>
                  <div style="color:#718096;font-size:12px;margin-top:4px;">마지막 봇 실행: {saved.get('updated','?')}</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background:linear-gradient(135deg,#2a2010 0%,#16213e 100%);border:1px solid #ff9800;border-radius:14px;padding:14px 20px;margin-bottom:14px;">
                  <span style="font-size:15px;font-weight:700;color:#ffb74d;">🟡 {rg} — 현금 대기</span>
                  <div style="color:#718096;font-size:12px;margin-top:4px;">마지막 봇 실행: {saved.get('updated','?')}</div></div>""", unsafe_allow_html=True)

            # A트랙 카드
            st.markdown("""<div style="display:flex;align-items:center;margin:4px 0 10px 0;">
              <div style="background:#0f3460;border-radius:8px;padding:4px 14px;font-size:14px;font-weight:700;color:#63b3ed;">🏆 A트랙 — 신선한 신고가 스윙</div>
              <div style="color:#4a5568;font-size:12px;margin-left:10px;">15일 보유 · 종가 -10% 손절</div></div>""", unsafe_allow_html=True)
            cands = saved.get("candidates", [])
            newset = set(saved.get("new_entries", []))
            if cands:
                for c in cands:
                    isbuy = c["code"] in newset
                    buy_pill = '<span class="pill" style="background:#00c853;color:#08210f;">🎯 오늘 매수</span>' if isbuy else ''
                    earn_pill = ''
                    if c.get("earn") == "GOOD":
                        yy = c.get("earn_yoy")
                        earn_pill = f'<span class="pill" style="background:#3b2a06;color:#ffc107;">🚀 실적 {"+%.0f%%" % yy if yy is not None else "흑자전환"} · 비중1.5x</span>'
                    st.markdown(
                        f'<div class="cand-card{" buy" if isbuy else ""}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div><span class="cand-name">{c["name"]}</span><span class="cand-code">{c["code"]}</span>{buy_pill}{earn_pill}</div>'
                        f'<div style="font-size:17px;font-weight:700;color:#e2e8f0;">{c["close"]:,.0f}<span style="font-size:12px;color:#718096;">원</span></div>'
                        f'</div>'
                        f'<div><span class="metric">52주고가 <b>{c["d52"]:+.1f}%</b></span>'
                        f'<span class="metric">거래대금 <b>{c["avg_value_억"]:,.0f}억</b></span>'
                        f'<span class="metric">매수가 <b>내일 시가</b> (갭+2%↑ 보류)</span></div>'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.markdown("""<div class="cand-card" style="text-align:center;color:#718096;padding:22px;">
                  오늘은 조건 통과 종목이 없습니다 — 신고가권이 마른 조정 구간엔 쉬는 것도 전략입니다 😌</div>""", unsafe_allow_html=True)

            # B트랙 카드
            cands_b = saved.get("candidates_b", [])
            if cands_b:
                st.markdown("""<div style="display:flex;align-items:center;margin:18px 0 10px 0;">
                  <div style="background:#123a5c;border-radius:8px;padding:4px 14px;font-size:14px;font-weight:700;color:#7cc7ff;">🔵 B트랙 — 초대형 과매도 회귀</div>
                  <div style="color:#4a5568;font-size:12px;margin-left:10px;">+5% 지정가 익절 · 15일 · 승률 ~73%</div></div>""", unsafe_allow_html=True)
                newb = set(saved.get("new_entries_b", []))
                for c in cands_b:
                    isbuy = c["code"] in newb
                    buy_pill = '<span class="pill" style="background:#2d6cdf;color:#eaf3ff;">🔵 오늘 매수</span>' if isbuy else ''
                    flow = f"{c['flow20']:+.1f}%" if c.get("flow20") is not None else "N/A"
                    flow_col = "#4ade80" if (c.get("flow20") or 0) > 0 else "#ff8a80"
                    st.markdown(
                        f'<div class="cand-card{" buyb" if isbuy else ""}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div><span class="cand-name">{c["name"]}</span><span class="cand-code">{c["code"]}</span>{buy_pill}</div>'
                        f'<div style="font-size:17px;font-weight:700;color:#e2e8f0;">{c["close"]:,.0f}<span style="font-size:12px;color:#718096;">원</span></div>'
                        f'</div>'
                        f'<div><span class="metric">RSI(2) <b>{c["rsi2"]}</b></span>'
                        f'<span class="metric">외인 20일수급 <b style="color:{flow_col};">{flow}</b></span>'
                        f'<span class="metric">200일선 <b>{c["ma200_dist"]:+.1f}%</b></span>'
                        f'<span class="metric">목표 <b style="color:#4ade80;">{c["close"]*1.05:,.0f}원</b></span></div>'
                        f'</div>', unsafe_allow_html=True)
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
                is_b = p.get("track") == "B"
                max_d = 15 if is_b else HOLD_DAYS
                barw = min(int(held / max_d * 100), 100)
                stop_txt = (f"목표 {p['target_price']:,.0f}원" if is_b and p.get("target_price")
                            else f"{p['stop_price']:,.0f}원" if p.get("stop_price") else "-")
                ptag = '<span class="tag" style="background:#4a3800;color:#ffc107;margin-left:8px;">체결대기</span>' if pending else ""
                if is_b:
                    ptag += '<span class="tag" style="background:#123a5c;color:#7cc7ff;margin-left:8px;">B 회귀</span>'
                cur_txt = f"{cur:,.0f}원" if cur else "-"
                ret_txt = f"{ret:+.1f}%" if ret is not None else "-"
                entry_txt = "대기" if pending else f"{p['entry_price']:,.0f}원"
                st.markdown(
                    f'<div class="stock-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
                    f'<div><span class="stock-name">{p["name"]}</span><span class="stock-code">{p["code"]}</span>{ptag}</div>'
                    f'<div style="text-align:right;"><div style="color:#718096;font-size:11px;">현재가</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{pct_color(ret)};">{cur_txt} <span style="font-size:13px;">({ret_txt})</span></div></div>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">'
                    f'<div class="mini-box"><div class="mini-title">진입 ({p["entry_date"][5:]})</div>'
                    f'<div style="color:#e2e8f0;font-size:14px;font-weight:600;">{entry_txt}</div></div>'
                    f'<div class="mini-box"><div class="mini-title">보유일 {held}/{max_d}</div>'
                    f'<div style="background:#2d3748;border-radius:4px;height:8px;margin-top:6px;"><div style="background:#63b3ed;width:{barw}%;height:8px;border-radius:4px;"></div></div></div>'
                    f'<div class="mini-box"><div class="mini-title">{"목표가 (+5%)" if is_b else "손절가 (-10%)"}</div>'
                    f'<div style="color:{"#00c853" if is_b else "#ff1744"};font-size:14px;font-weight:600;">{stop_txt}</div></div>'
                    f'</div></div>', unsafe_allow_html=True)
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
**국면 게이트** — 코스피 종가 vs 120일선, **5일 연속 유지 시에만 ON/OFF 전환**(요동 방지 히스테리시스). ON일 때만 매매, OFF면 전량 현금

**매수 조건 (전부 충족 — v14.4 밴드 정밀화)**
1. 시가총액 1,000억 ~ 5조 / 20일 평균 거래대금 30억 이상
2. **52주 최고가 대비 -5% ~ -1% 밴드** (딱 붙은 종목은 제외 — 밴드분석 결과)
3. **MA5/MA20 이격 +4~8%** (추세 미형성·과열 제외)
4. **20일 수익률 +10~25%** (모멘텀 스윗스팟)
5. 당일 거래량이 5일평균 4배 이상 폭증한 날은 제외 (0/5년 회피밴드)
6. **52주 최고가가 최근 5거래일 내 갱신**된 신선한 고점만 (6일+ 묵은 고점은 엣지 0 — v14.5)
7. **영업이익 역성장(YoY -10%↓) 기업 제외** + 고성장(+50%↑)·흑자전환 기업 우선 랭킹 (v14.7 — DART 시점정합 5년 검증: 역성장 신고가는 4년 연속 마이너스, 폭발성장 신고가는 5/5년 플러스)
5. 진입일 시가 갭이 **+2% 이상(추격금지) 또는 -3% 이하(급락출발)** 면 매수 취소

**서킷브레이커 (손절 폭포 방어)**
- 최근 청산 10건 중 손절이 7건 이상이면 → **5거래일 신규진입 중단** (보유종목 관리는 계속)

**청산 (조기 익절 없음 — 승자 태우기)**
- {HOLD_DAYS}거래일 만기 매도 or **종가 기준 -10% 손절** (장중 꼬리에 안 잘림)

**자금 운용** — A트랙 최대 {SLOTS}종목 분산 · **🚀실적성장 신호는 1.5배 비중** (확신가중, v14.8 — 전 슬롯구성 CAGR +3~6%p 검증. 2배는 특정연도 쏠림으로 기각)

**🔵 보조 B트랙 (조정장 회귀 · 승률 74% 검증)**
- 초대형주(시총 5조↑) + 200일선 위 + RSI(2) < 10 과매도 → 익일 시가 매수
- 청산: **+5% 지정가 익절** or 15거래일 종가 매도 (손절 없음) · 최대 3슬롯 (+3%→+5% 상향: 승률 78→69%, 거래당 수익 1.7배 — 3/3년 검증)
- **수급 필터**: 외인 20일 누적 순매도가 거래대금 -10% 이하인 종목 제외 (4/4년 검증 회피신호) · 후보는 외인 매집 강도순 랭킹
- A트랙이 조용한 조정기에 신호가 나는 상호보완 구조

**검증 근거 (5년 백테스트, 2021.7~2026.7, 수수료 0.3% 반영)**
- 30개 기술 팩터 전수조사 중 유일하게 5년 전 연도에서 무작위 대비 우위를 유지한 팩터
- 학계 검증된 '52주 신고가 모멘텀' 이상현상과 일치 (George & Hwang 2004)
- 완전분산 기준 CAGR +10~18%, MDD -19~-31%, **전 연도 플러스** (2022 +9%, 2024 +12%)
- 현실 기대치: 12슬롯 기준 **CAGR 9~15%** (생존편향·미래 불확실성 감안해 보수적으로)

⚠️ 백테스트는 미래를 보장하지 않습니다. 실계좌 전 모의운용 권장.
        """)
