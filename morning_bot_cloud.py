# -*- coding: utf-8 -*-
"""
주식봇 v14 — 52주 신고가 스윙 (매일 아침 7:10 KST, GitHub Actions)

전략 (5년 백테스트 전 연도 플러스 검증):
  국면: 코스피 종가 > 120일선  → 아니면 전량 현금 대기
  진입(v14.4 밴드정밀화): 시총 1000억~5조 + 거래대금 30억↑
        + 52주고가 -5~-1%(최근 5일내 갱신된 신선한 고점만) + 이격 4~8% + 20일수익 10~25% + 폭증일 제외
        (시가 갭 +2% 이상이면 다음날 자동 진입취소)
  청산: 15거래일 보유 or 종가 -10% 손절 (조기 익절 없음)
  분산: 최대 12슬롯 균등 — 봇이 가상 포트폴리오로 추적, 매도알림 발송
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os, json
import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

GMAIL_ADDRESS     = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PW      = os.environ["GMAIL_APP_PW"]
RECEIVE_EMAIL     = os.environ["RECEIVE_EMAIL"]

# ── v14 전략 상수 ─────────────────────────
SLOTS             = 12                  # 최대 동시보유
HOLD_DAYS         = 15                  # 보유 거래일
STOP_LOSS         = 0.10                # 손절 -10% (종가 기준 — 장중 꼬리에 안 잘림, 승률 34→41% 검증)
NEAR_HIGH         = -5.0                # 신고가 밴드 하한 (-5%)
NEAR_HIGH_TOP     = -1.0                # 신고가 밴드 상한 (-1%: 딱 붙은 종목 제외 — 밴드분석 v14.4)
PREM_MIN, PREM_MAX   = 4.0, 8.0         # MA5/MA20 이격 4~8% (추세 형성 구간만)
RET20_MIN, RET20_MAX = 10.0, 25.0       # 진입 전 20일 수익 10~25% (과열/미지근 제외)
VOL_SPIKE_MAX     = 4.0                 # 당일 거래량 5일평균 4배↑ 폭증일 제외
FRESH_DAYS        = 5                   # 52주 최고가가 최근 5거래일 내 갱신된 종목만 (신선도 — v14.5)
EARN_AVOID_YOY    = -10.0               # 최신 분기 영업이익 YoY -10% 이하(역성장) 제외 — v14.7, 5/5년 검증
EARN_GOOD_YOY     = 50.0                # YoY +50%↑(고성장) 또는 흑자전환 → 랭킹 우선
DART_API_KEY      = os.environ.get("DART_API_KEY", "")   # 없으면 실적 필터 생략(경고만)
GAP_MAX           = 2.0                 # 시가 갭 +2% 이상이면 진입취소
GAP_MIN           = -3.0                # 시가 갭 -3% 이하(급락 출발)도 진입취소
BREAKER_WINDOW    = 10                  # 서킷브레이커: 최근 청산 10건 중
BREAKER_STOPS     = 7                   #   손절이 7건 이상이면
BREAKER_PAUSE     = 5                   #   5거래일 신규진입 중단
MARKET_CAP_MIN    = 100_000_000_000
MARKET_CAP_MAX    = 5_000_000_000_000
MIN_TRADING_VALUE = 3_000_000_000       # 거래대금 30억
REGIME_MA         = 120                 # 코스피 국면 이평
REGIME_CONFIRM    = 5                   # 국면 전환 확인일수 (5일 연속 유지 시 전환 — 요동 방지)
HISTORY_PATH      = "history.json"

# ── 보조 트랙 B: 초대형주 회귀 (조정장용 · 승률 74% 검증) ──
B_SLOTS           = 3                   # 보조 트랙 슬롯
B_MARCAP_MIN      = 5_000_000_000_000   # 시총 5조↑ (초대형)
B_RSI2_MAX        = 10                  # RSI(2) 과매도
B_TARGET          = 0.03                # +3% 목표 익절
B_HOLD            = 10                  # 최대 10거래일 (목표 미달 시 종가 청산)
B_FLOW_AVOID      = -10.0               # 외인 20일 누적 순매도 ≤ 거래대금의 -10% → 제외 (4/4년 검증 회피신호)


# ─────────────────────────────────────────
def check_us_market():
    """미국 시장 현황 (참고용)"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        sp  = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze().pct_change()
        nq  = yf.download("^IXIC", start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze().pct_change()
        sox = yf.download("^SOX",  start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze().pct_change()
        return float(sp.iloc[-1]), float(nq.iloc[-1]), float(sox.iloc[-1]), sp.index[-1].strftime("%Y-%m-%d")
    except:
        return 0, 0, 0, "N/A"


def check_regime():
    """국면: 코스피 vs 120일선 + 히스테리시스(5일 연속 유지 시에만 전환 — 요동 방지)"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=420)).strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader("KS11", start, end)
        close = df["Close"]
        if len(close) < REGIME_MA + 30:
            return False, "코스피 데이터 부족"
        ma  = close.rolling(REGIME_MA).mean()
        raw = (close > ma).dropna().tolist()
        state = raw[0]; cnt = 0
        for x in raw[1:]:
            if x != state:
                cnt += 1
                if cnt >= REGIME_CONFIRM:
                    state = x; cnt = 0
            else:
                cnt = 0
        c    = float(close.iloc[-1])
        m    = float(ma.iloc[-1])
        dist = (c / m - 1) * 100
        pend = f" (전환 진행 {cnt}/{REGIME_CONFIRM}일)" if cnt > 0 else ""
        if state:
            return True, f"코스피 {c:,.0f} vs 120일선 {m:,.0f} ({dist:+.1f}%) — 매매 ON{pend}"
        return False, f"코스피 {c:,.0f} vs 120일선 {m:,.0f} ({dist:+.1f}%) — 현금 대기{pend}"
    except Exception as e:
        return False, f"국면 확인 실패: {e}"


# ─────────────────────────────────────────
# 가상 포트폴리오 (history.json)
# 형식: {"format":"v14","positions":[...],"closed":[...],"legacy":[구버전 기록]}
# position: {code,name,entry_date,entry_price(없으면 pending),stop_price,ref_close}
# closed  : {code,name,entry_date,entry_price,exit_date,exit_price,ret_pct,reason}
# ─────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            h = json.load(f)
        if isinstance(h, list):                      # v13 이하 → 마이그레이션
            print("history.json v13 형식 감지 → v14로 마이그레이션 (기존 기록은 legacy 보존)")
            return {"format": "v14", "positions": [], "closed": [], "legacy": h}
        h.setdefault("positions", []); h.setdefault("closed", []); h.setdefault("legacy", [])
        return h
    return {"format": "v14", "positions": [], "closed": [], "legacy": []}


def save_history(h):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)


def update_positions(hist):
    """보유 포지션 갱신: pending 체결확정/갭취소 → 손절 체크 → 만기 체크"""
    today_str = datetime.today().strftime("%Y-%m-%d")
    kept, sell_alerts, cancels = [], [], []

    for p in hist["positions"]:
        try:
            start = (pd.Timestamp(p["entry_date"]) - timedelta(days=7)).strftime("%Y-%m-%d")
            df = fdr.DataReader(p["code"], start)
            if df is None or df.empty:
                kept.append(p); continue
            df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")

            track = p.get("track", "A")

            # 1) pending 체결 확정 (진입일 시가 = 체결가; A트랙만 갭 ±필터로 취소)
            if p.get("entry_price") is None:
                if p["entry_date"] in df.index:
                    o = float(df.loc[p["entry_date"], "Open"])
                    gap = (o - p["ref_close"]) / p["ref_close"] * 100 if p.get("ref_close") else 0
                    if track == "A" and (gap >= GAP_MAX or gap <= GAP_MIN):
                        cancels.append({**p, "reason": f"갭 {gap:+.1f}% 진입취소"})
                        continue
                    p["entry_price"] = round(o, 2)
                    if track == "A":
                        p["stop_price"] = round(o * (1 - STOP_LOSS), 2)
                    else:
                        p["target_price"] = round(o * (1 + B_TARGET), 2)
                else:
                    kept.append(p); continue    # 아직 진입일 시세 없음 (휴장 등)

            held = df[df.index >= p["entry_date"]]
            if held.empty:
                kept.append(p); continue
            days_held = len(held)
            cur = float(held["Close"].iloc[-1])
            ret = (cur - p["entry_price"]) / p["entry_price"] * 100

            if track == "B":
                # B-1) 목표 +3% 터치 → 익절 (지정가 체결 가정)
                hit = held[held["High"] >= p["target_price"]]
                if not hit.empty:
                    hist["closed"].append({
                        "code": p["code"], "name": p["name"], "track": "B",
                        "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                        "exit_date": hit.index[0], "exit_price": p["target_price"],
                        "ret_pct": round(B_TARGET * 100, 2), "reason": f"목표 +{B_TARGET*100:.0f}%",
                    })
                    sell_alerts.append({"name": p["name"], "code": p["code"],
                                        "msg": f"🎯 [B트랙] 목표가 {p['target_price']:,.0f}원 도달 ({hit.index[0]}) — 지정가 미체결시 오늘 매도"})
                    continue
                # B-2) 10거래일 만기
                if days_held >= B_HOLD:
                    hist["closed"].append({
                        "code": p["code"], "name": p["name"], "track": "B",
                        "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                        "exit_date": held.index[-1], "exit_price": cur,
                        "ret_pct": round(ret, 2), "reason": f"{B_HOLD}일 만기(B)",
                    })
                    sell_alerts.append({"name": p["name"], "code": p["code"],
                                        "msg": f"⏰ [B트랙] {B_HOLD}거래일 만기 ({ret:+.1f}%) — 오늘 매도"})
                    continue
            else:
                # 2) A: 손절 — 종가 기준 -10% 이탈 (장중 꼬리 무시)
                breach = held[held["Close"] <= p["stop_price"]]
                if not breach.empty:
                    breach_date  = breach.index[0]
                    breach_close = float(breach["Close"].iloc[0])
                    ret_stop = (breach_close - p["entry_price"]) / p["entry_price"] * 100
                    hist["closed"].append({
                        "code": p["code"], "name": p["name"], "track": "A",
                        "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                        "exit_date": breach_date, "exit_price": breach_close,
                        "ret_pct": round(ret_stop, 2), "reason": "손절 -10%(종가)",
                    })
                    sell_alerts.append({"name": p["name"], "code": p["code"],
                                        "msg": f"🛑 종가가 손절선 {p['stop_price']:,.0f}원 이탈 ({breach_date}, {ret_stop:+.1f}%) — 오늘 매도"})
                    continue

                # 3) A: 만기 — 15거래일 도래
                if days_held >= HOLD_DAYS:
                    hist["closed"].append({
                        "code": p["code"], "name": p["name"], "track": "A",
                        "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                        "exit_date": held.index[-1], "exit_price": cur,
                        "ret_pct": round(ret, 2), "reason": f"{HOLD_DAYS}일 만기",
                    })
                    sell_alerts.append({"name": p["name"], "code": p["code"],
                                        "msg": f"⏰ {HOLD_DAYS}거래일 만기 (수익 {ret:+.1f}%) — 오늘 매도"})
                    continue

            # 4) 계속 보유
            p["current"]   = round(cur, 2)
            p["ret_pct"]   = round(ret, 2)
            p["days_held"] = days_held
            kept.append(p)
        except Exception as e:
            print(f"  포지션 갱신 실패 {p.get('name')}: {e}")
            kept.append(p)

    hist["positions"] = kept
    return sell_alerts, cancels


def _rsi(close, p=2):
    d = close.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/p, adjust=False).mean(); rd = dn.ewm(alpha=1/p, adjust=False).mean()
    return (100 - 100 / (1 + ru / rd.replace(0, np.nan))).fillna(50)


def _foreign_flow20(ticker, close_price, avg_value):
    """네이버 수급 2페이지(≈40일)로 외인 20일 누적 순매수대금 강도(%) 계산.
    강도 = 20일 누적 순매수대금 / (20일평균 거래대금 × 20). 실패 시 None."""
    try:
        import requests
        from io import StringIO
        shares = []
        for page in (1, 2):
            url = f"https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            for t in pd.read_html(StringIO(r.text)):
                cols = str(t.columns.tolist())
                if "외국인" in cols and "기관" in cols:
                    t.columns = ["_".join(c) if isinstance(c, tuple) else c for c in t.columns]
                    dc = next((c for c in t.columns if "날짜" in c), None)
                    fc = next((c for c in t.columns if "외국인" in c and "순매" in c), None)
                    t = t.dropna(subset=[dc])
                    shares += [float(x) if pd.notna(x) else 0.0 for x in t[fc]]
                    break
        if len(shares) < 20 or not avg_value:
            return None
        return sum(shares[:20]) * close_price / (avg_value * 20) * 100
    except Exception:
        return None


def get_candidates_b(exclude_codes):
    """트랙B 스캔: 초대형주(5조↑) + 200일선 위 + RSI2<10 과매도
    + 외인 20일 수급 필터/랭킹 (강매도 제외 · 매집 우선 — 2.9년 검증)"""
    print("트랙B(초대형 회귀) 스캔 중...")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    all_s = pd.concat([fdr.StockListing("KOSPI"), fdr.StockListing("KOSDAQ")], ignore_index=True)
    mega  = all_s[all_s["Marcap"] >= B_MARCAP_MIN]
    name_map = dict(zip(mega["Code"], mega["Name"]))
    results = []
    for tk in mega["Code"].tolist():
        if tk in exclude_codes: continue
        try:
            df = fdr.DataReader(tk, start)
            if df.empty or len(df) < 220: continue
            close = df["Close"]
            c     = float(close.iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])
            if np.isnan(ma200) or c <= ma200: continue
            r2 = float(_rsi(close, 2).iloc[-1])
            if r2 >= B_RSI2_MAX: continue
            avg_value = float((df["Volume"] * close).rolling(20).mean().iloc[-1])
            results.append({"code": tk, "name": name_map.get(tk, tk),
                            "close": c, "rsi2": round(r2, 1),
                            "ma200_dist": round((c/ma200-1)*100, 1),
                            "avg_value": avg_value})
        except:
            continue
    # 통과 후보만 수급 조회 (건수 적어 부담 없음)
    kept = []
    for r in results:
        f20 = _foreign_flow20(r["code"], r["close"], r["avg_value"])
        r["flow20"] = round(f20, 1) if f20 is not None else None
        if f20 is not None and f20 <= B_FLOW_AVOID:
            print(f"   제외(외인 강매도 {f20:+.1f}%): {r['name']}")
            continue
        kept.append(r)
    # 랭킹: 외인 20일 누적 순매수 강도 높은 순 (수급 미확인은 후순위)
    kept.sort(key=lambda x: -(x["flow20"] if x["flow20"] is not None else -99))
    print(f"   -> B후보 {len(kept)}개 (수급필터 후)")
    return kept


# ─────────────────────────────────────────
def get_earnings_states(stock_codes):
    """후보 종목들의 최신 공시 분기 영업이익 상태 조회 (DART 다중회사 API).
    반환 {code: ("BAD"|"GOOD"|"NEUTRAL", yoy%)}. 키 없거나 실패 시 빈 dict."""
    if not DART_API_KEY or not stock_codes:
        if not DART_API_KEY: print("   (DART_API_KEY 없음 → 실적 필터 생략)")
        return {}
    try:
        import requests, pickle
        mp = pickle.load(open("dart_corpmap.pkl", "rb"))       # corp→stock
        rev = {v: k for k, v in mp.items()}
        corps = {rev[c]: c for c in stock_codes if c in rev}   # corp→stock
        if not corps: return {}
        # 최신 보고서부터 역순으로 시도 (연도, 보고서코드)
        today = datetime.today()
        tries = []
        for y in (today.year, today.year - 1):
            for rc in ("11014", "11012", "11013", "11011"):
                tries.append((str(y), rc))
        tries.sort(key=lambda x: (x[0], {"11011":"4","11013":"1","11012":"2","11014":"3"}[x[1]]), reverse=True)
        out = {}
        for y, rc in tries:
            if len(out) >= len(corps): break
            r = requests.get("https://opendart.fss.or.kr/api/fnlttMultiAcnt.json",
                params=dict(crtfc_key=DART_API_KEY, corp_code=",".join(corps.keys()),
                            bsns_year=y, reprt_code=rc), timeout=20)
            d = r.json()
            if d.get("status") != "000": continue
            best = {}
            for x in d["list"]:
                if x.get("account_nm") != "영업이익": continue
                sc = corps.get(x["corp_code"])
                if sc is None or sc in out: continue
                if sc in best and x.get("fs_div") != "CFS": continue
                best[sc] = x
            for sc, x in best.items():
                def num(v):
                    try: return float(str(v).replace(",", ""))
                    except: return None
                t, f = num(x.get("thstrm_amount")), num(x.get("frmtrm_amount"))
                if t is None: continue
                if f is not None and f > 0 and t > 0:
                    yoy = (t / f - 1) * 100
                    out[sc] = ("BAD" if yoy < EARN_AVOID_YOY else
                               "GOOD" if yoy >= EARN_GOOD_YOY else "NEUTRAL", round(yoy, 1))
                elif f is not None and f <= 0 < t:
                    out[sc] = ("GOOD", None)      # 흑자전환
                else:
                    out[sc] = ("NEUTRAL", None)   # 적자 등 → 중립
        return out
    except Exception as e:
        print(f"   실적 조회 실패(생략): {e}")
        return {}


def get_candidates(exclude_codes):
    """v14 스캔: 52주 신고가 -5% 이내 + 거래대금 30억 + MA5>MA20"""
    print("종목 스캔 중...")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    kospi  = fdr.StockListing("KOSPI");  kospi["market"]  = "KOSPI"
    kosdaq = fdr.StockListing("KOSDAQ"); kosdaq["market"] = "KOSDAQ"
    all_s  = pd.concat([kospi, kosdaq], ignore_index=True)
    filt   = all_s[(all_s["Marcap"] >= MARKET_CAP_MIN) & (all_s["Marcap"] <= MARKET_CAP_MAX)]
    name_map = dict(zip(filt["Code"], filt["Name"]))
    tickers  = [t for t in filt["Code"].tolist() if t not in exclude_codes]
    print(f"   대상 {len(tickers)}개")

    results = []
    for i, tk in enumerate(tickers):
        if i % 200 == 0:
            print(f"   {i}/{len(tickers)}..."); sys.stdout.flush()
        try:
            df = fdr.DataReader(tk, start)
            if df.empty or len(df) < 260:
                continue
            close = df["Close"]; high = df["High"]; vol = df["Volume"]
            c     = float(close.iloc[-1])
            h52   = float(high.rolling(252).max().iloc[-1])
            if h52 <= 0: continue
            d52   = (c / h52 - 1) * 100
            if not (NEAR_HIGH <= d52 < NEAR_HIGH_TOP): continue   # v14.4 신고가 밴드 -5~-1%
            avg_value = float((vol * close).rolling(20).mean().iloc[-1])
            if np.isnan(avg_value) or avg_value < MIN_TRADING_VALUE: continue
            ma5  = float(close.rolling(5).mean().iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            if np.isnan(ma20) or ma20 <= 0: continue
            prem = (ma5 / ma20 - 1) * 100
            if not (PREM_MIN <= prem < PREM_MAX): continue    # v14.4 추세이격 밴드
            ret20 = (c - float(close.iloc[-21])) / float(close.iloc[-21]) * 100 if len(close) > 21 else 0
            if not (RET20_MIN <= ret20 < RET20_MAX): continue # v14.4 모멘텀 밴드
            vol5 = float(vol.iloc[-6:-1].mean())
            if vol5 > 0 and float(vol.iloc[-1]) / vol5 >= VOL_SPIKE_MAX: continue  # 폭증일 제외
            if float(high.iloc[-FRESH_DAYS:].max()) < h52 * 0.9999: continue      # v14.5 신선도: 5일내 신고가 갱신
            results.append({
                "code": tk, "name": name_map.get(tk, tk),
                "close": c, "hi52": h52, "d52": round(d52, 2),
                "avg_value_억": round(avg_value / 100_000_000, 1),
            })
        except:
            continue

    # v14.7 실적 오버레이: 역성장 제외 + 고성장/흑자전환 우선
    earn = get_earnings_states([r["code"] for r in results])
    kept = []
    for r in results:
        st_, yoy = earn.get(r["code"], ("NEUTRAL", None))
        if st_ == "BAD":
            print(f"   제외(영업이익 역성장 {yoy}%): {r['name']}")
            continue
        r["earn"] = st_; r["earn_yoy"] = yoy
        kept.append(r)
    kept.sort(key=lambda x: (0 if x.get("earn") == "GOOD" else 1, -x["d52"]))
    print(f"   -> 후보 {len(kept)}개 (실적필터 후)")
    return kept


# ─────────────────────────────────────────
def build_email(regime_on, regime_msg, sp_ret, nq_ret, sox_ret, us_date,
                positions, sell_alerts, cancels, candidates, new_entries,
                new_entries_b=None):
    today = datetime.today().strftime("%Y-%m-%d")
    g = "#00c853"; r = "#ff1744"

    # 매도 알림
    sell_html = ""
    if sell_alerts:
        items = "".join(f"<li style='margin:4px 0;'><b>{a['name']}</b> ({a['code']}) — {a['msg']}</li>" for a in sell_alerts)
        sell_html = f"""<div style="background:#2a1a1a;border:1px solid #ff1744;padding:14px;border-radius:8px;margin-bottom:14px;">
          <h3 style="color:{r};margin:0 0 8px;">📤 오늘 매도</h3><ul style="margin:0;padding-left:18px;color:#eee;">{items}</ul></div>"""
    if cancels:
        items = "".join(f"<li style='margin:4px 0;'><b>{c['name']}</b> — {c['reason']}</li>" for c in cancels)
        sell_html += f"""<div style="background:#241a2a;padding:10px 14px;border-radius:8px;margin-bottom:14px;color:#caa;font-size:13px;">
          갭 진입취소: <ul style="margin:4px 0 0;padding-left:18px;">{items}</ul></div>"""

    # 보유 포지션 (A/B 트랙 표시)
    pos_rows = ""
    for p in sorted(positions, key=lambda x: (x.get("track","A"), -(x.get("days_held",0) or 0))):
        pending = p.get("entry_price") is None
        ret = p.get("ret_pct"); col = g if (ret or 0) >= 0 else r
        track = p.get("track", "A")
        max_d = HOLD_DAYS if track == "A" else B_HOLD
        exit_txt = (f"{p.get('stop_price',0):,.0f}" if track == "A" else
                    f"목표 {p.get('target_price',0):,.0f}") if not pending else "-"
        badge = "" if track == "A" else " <span style='background:#123a5c;color:#7cc7ff;border-radius:4px;padding:1px 6px;font-size:10px;'>B</span>"
        pos_rows += f"""<tr>
          <td style="padding:8px;">{p['name']}{badge}<span style="color:#666;font-size:11px;"> {p['code']}</span></td>
          <td style="padding:8px;text-align:center;">{p['entry_date'][5:]}</td>
          <td style="padding:8px;text-align:right;">{'체결대기' if pending else f"{p['entry_price']:,.0f}"}</td>
          <td style="padding:8px;text-align:right;">{f"{p.get('current',0):,.0f}" if not pending else '-'}</td>
          <td style="padding:8px;text-align:right;color:{col};font-weight:bold;">{f"{ret:+.1f}%" if ret is not None else '-'}</td>
          <td style="padding:8px;text-align:center;">{p.get('days_held','-')}/{max_d}일</td>
          <td style="padding:8px;text-align:right;color:{r if track=='A' else g};">{exit_txt}</td>
        </tr>"""
    n_a = sum(1 for p in positions if p.get("track","A")=="A")
    n_b = len(positions) - n_a
    pos_html = f"""<div style="background:#1e1e1e;padding:14px;border-radius:8px;margin-bottom:14px;">
      <h3 style="color:#64b5f6;margin:0 0 10px;">💼 보유 — A(신고가) {n_a}/{SLOTS} · B(초대형회귀) {n_b}/{B_SLOTS}</h3>
      <table style="width:100%;border-collapse:collapse;font-size:13px;color:#ddd;">
        <tr style="color:#888;font-size:12px;"><th style="text-align:left;padding:8px;">종목</th><th>진입일</th>
        <th style="text-align:right;">진입가</th><th style="text-align:right;">현재가</th>
        <th style="text-align:right;">수익률</th><th>보유일</th><th style="text-align:right;">손절/목표</th></tr>
        {pos_rows if pos_rows else '<tr><td colspan="7" style="padding:12px;color:#666;">보유 없음</td></tr>'}
      </table></div>"""

    # 신규 매수
    buy_html = ""
    if new_entries:
        rows = "".join(f"""<tr>
          <td style="padding:8px;font-weight:bold;">{i}. {c['name']}<span style="color:#666;font-size:11px;"> {c['code']}</span></td>
          <td style="padding:8px;text-align:right;">{c['close']:,.0f}원</td>
          <td style="padding:8px;text-align:right;color:{g};">신고가 {c['d52']:+.1f}%{' · 🚀실적' if c.get('earn')=='GOOD' else ''}</td>
          <td style="padding:8px;text-align:right;">{c['avg_value_억']:,.0f}억</td>
        </tr>""" for i, c in enumerate(new_entries, 1))
        buy_html = f"""<div style="background:#1a2a1a;border:1px solid #00c853;padding:14px;border-radius:8px;margin-bottom:14px;">
          <h3 style="color:{g};margin:0 0 8px;">🎯 오늘 매수 (시가) — {len(new_entries)}종목</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px;color:#ddd;">{rows}</table>
          <p style="color:#888;font-size:12px;margin:8px 0 0;">※ 시가가 전일종가 대비 +{GAP_MAX:.0f}% 이상 갭상승이면 매수 보류 (봇도 자동 취소 처리)</p></div>"""
    elif regime_on:
        buy_html = """<div style="background:#1e1e1e;padding:12px 14px;border-radius:8px;margin-bottom:14px;color:#888;font-size:13px;">
          오늘 A트랙 신규 매수 없음 (슬롯 가득·후보 없음·브레이커)</div>"""

    # 트랙B 신규 매수 (초대형주 회귀)
    if new_entries_b:
        rows = "".join(f"""<tr>
          <td style="padding:8px;font-weight:bold;">{i}. {c['name']}<span style="color:#666;font-size:11px;"> {c['code']}</span></td>
          <td style="padding:8px;text-align:right;">{c['close']:,.0f}원</td>
          <td style="padding:8px;text-align:right;color:#7cc7ff;">RSI2 {c['rsi2']} · 외인20일 {('%+.1f%%' % c['flow20']) if c.get('flow20') is not None else 'N/A'}</td>
          <td style="padding:8px;text-align:right;color:{g};">목표 {c['close']*(1+B_TARGET):,.0f}원</td>
        </tr>""" for i, c in enumerate(new_entries_b, 1))
        buy_html += f"""<div style="background:#12233a;border:1px solid #2d6cdf;padding:14px;border-radius:8px;margin-bottom:14px;">
          <h3 style="color:#7cc7ff;margin:0 0 8px;">🔵 B트랙 매수 (초대형 과매도 회귀) — {len(new_entries_b)}종목</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px;color:#ddd;">{rows}</table>
          <p style="color:#889;font-size:12px;margin:8px 0 0;">※ 내일 시가 매수 후 <b>+{B_TARGET*100:.0f}% 지정가 매도</b> 예약 · 미체결 시 {B_HOLD}거래일째 종가 매도 · 손절 없음(승률 74% 검증) · 외인 20일 강매도 종목 자동제외, 매집순 랭킹</p></div>"""

    # 대기 후보
    watch_html = ""
    extra = [c for c in candidates if c["code"] not in {e["code"] for e in new_entries}][:5]
    if extra:
        rows = "".join(f"<li>{c['name']} — 신고가 {c['d52']:+.1f}% / {c['avg_value_억']:,.0f}억</li>" for c in extra)
        watch_html = f"""<div style="background:#1e1e1e;padding:12px 14px;border-radius:8px;margin-bottom:14px;font-size:12px;color:#999;">
          👀 대기 후보 (슬롯 없음): <ul style="margin:4px 0 0;padding-left:18px;">{rows}</ul></div>"""

    regime_col = g if regime_on else "#ff9800"
    html = f"""
<html><body style="background:#0d0d0d;color:#fff;font-family:sans-serif;padding:20px;">
  <div style="max-width:680px;margin:auto;">
    <h2 style="color:#64b5f6;">📈 주식봇 v14 신고가 스윙 — {today}</h2>
    <div style="background:#1e1e1e;padding:14px;border-radius:8px;margin-bottom:14px;">
      <b style="color:{regime_col};">{'🟢' if regime_on else '🟡'} {regime_msg}</b><br>
      <span style="color:#888;font-size:12px;">미국({us_date}): S&P <span style="color:#{'00c853' if sp_ret>0 else 'ff1744'}">{sp_ret*100:+.2f}%</span>
      · 나스닥 <span style="color:#{'00c853' if nq_ret>0 else 'ff1744'}">{nq_ret*100:+.2f}%</span>
      · SOX <span style="color:#{'00c853' if sox_ret>0 else 'ff1744'}">{sox_ret*100:+.2f}%</span></span>
    </div>
    {sell_html}{buy_html}{pos_html}{watch_html}
    <p style="color:#555;font-size:11px;margin-top:18px;">
      A트랙(주력): 신고가 -5~-1%·이격4~8%·모멘텀10~25% (v14.4 밴드정밀화) · {HOLD_DAYS}일/종가-10%손절 · {SLOTS}슬롯 |
      B트랙(보조): 초대형 RSI2 과매도 회귀 · +{B_TARGET*100:.0f}%목표/{B_HOLD}일 · {B_SLOTS}슬롯<br>
      국면: 코스피 120일선 (5일 확인) · 서킷브레이커: 10청산 중 7손절 시 A트랙 5일 중단<br>
      5년 검증: A 전연도 플러스(CAGR 8~12% 기대) · B 승률 74% | 투자는 본인 판단 하에.
    </p>
  </div>
</body></html>"""
    return html


def send_email(subject, body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = RECEIVE_EMAIL
    msg.attach(MIMEText(body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PW)
        smtp.sendmail(GMAIL_ADDRESS, RECEIVE_EMAIL, msg.as_string())


# ─────────────────────────────────────────
def main():
    today_str = datetime.today().strftime("%Y-%m-%d")
    print(f"\n{'='*44}\n주식봇 v14 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*44}\n")

    sp_ret, nq_ret, sox_ret, us_date = check_us_market()
    regime_on, regime_msg = check_regime()
    print(f"국면: {regime_msg}")

    hist = load_history()

    # 1) 보유 포지션 갱신 (손절/만기/체결확정)
    sell_alerts, cancels = update_positions(hist)
    print(f"보유 {len(hist['positions'])}개 / 매도알림 {len(sell_alerts)} / 갭취소 {len(cancels)}")

    # 2) 서킷브레이커: 최근 청산 10건 중 손절 7건↑ → 5거래일 신규진입 중단
    brk = hist.setdefault("breaker", {"pause_left": 0, "reset_date": ""})
    closed_sorted = sorted(hist["closed"], key=lambda c: c.get("exit_date", ""))
    recent = [c for c in closed_sorted if c.get("exit_date", "") > brk.get("reset_date", "")][-BREAKER_WINDOW:]
    n_stops = sum(1 for c in recent if "손절" in c.get("reason", ""))
    if brk["pause_left"] == 0 and len(recent) >= BREAKER_WINDOW and n_stops >= BREAKER_STOPS:
        brk["pause_left"] = BREAKER_PAUSE
        brk["reset_date"] = today_str
        print(f"🚨 서킷브레이커 발동: 최근 {BREAKER_WINDOW}건 중 손절 {n_stops}건 → {BREAKER_PAUSE}일 신규중단")
    breaker_active = brk["pause_left"] > 0
    if breaker_active:
        brk["pause_left"] -= 1
        regime_msg += f" | 🚨 서킷브레이커: 신규진입 중단 (잔여 {brk['pause_left']+1}일)"

    # 3-A) 트랙A 스캔 (국면 ON + 브레이커 미발동)
    candidates, new_entries = [], []
    a_pos = [p for p in hist["positions"] if p.get("track", "A") == "A"]
    if regime_on and breaker_active:
        print("서킷브레이커 활성 → A트랙 스캔 생략 (보유종목 관리만)")
    elif regime_on:
        held_codes = {p["code"] for p in hist["positions"]}
        candidates = get_candidates(held_codes)
        empty = SLOTS - len(a_pos)
        for c in candidates[:max(0, empty)]:
            hist["positions"].append({
                "code": c["code"], "name": c["name"], "track": "A",
                "entry_date": today_str, "entry_price": None,
                "ref_close": c["close"], "stop_price": None,
            })
            new_entries.append(c)
        print(f"A트랙 신규 {len(new_entries)}개 (빈슬롯 {empty})")
    else:
        print("약세장 → A트랙 스캔 생략, 현금 대기")

    # 3-B) 트랙B 스캔 (국면 ON이면 브레이커와 무관 — 조정기 회귀는 이때가 제철)
    candidates_b, new_entries_b = [], []
    if regime_on:
        held_codes = {p["code"] for p in hist["positions"]}
        b_pos = [p for p in hist["positions"] if p.get("track") == "B"]
        candidates_b = get_candidates_b(held_codes)
        empty_b = B_SLOTS - len(b_pos)
        for c in candidates_b[:max(0, empty_b)]:
            hist["positions"].append({
                "code": c["code"], "name": c["name"], "track": "B",
                "entry_date": today_str, "entry_price": None,
                "ref_close": c["close"], "target_price": None,
            })
            new_entries_b.append(c)
        print(f"B트랙 신규 {len(new_entries_b)}개 (빈슬롯 {empty_b})")

    save_history(hist)

    # 4) candidates.json (Streamlit 앱용)
    with open("candidates.json", "w", encoding="utf-8") as f:
        json.dump({
            "format": "v14",
            "updated": datetime.today().strftime("%Y-%m-%d %H:%M"),
            "regime_on": regime_on, "regime_msg": regime_msg,
            "candidates": candidates[:20],
            "new_entries": [c["code"] for c in new_entries],
            "candidates_b": candidates_b[:10],
            "new_entries_b": [c["code"] for c in new_entries_b],
        }, f, ensure_ascii=False, indent=2)

    # 5) 메일
    body = build_email(regime_on, regime_msg, sp_ret, nq_ret, sox_ret, us_date,
                       hist["positions"], sell_alerts, cancels, candidates, new_entries,
                       new_entries_b)
    n_closed_today = len(sell_alerts)
    n_buy = len(new_entries) + len(new_entries_b)
    subject = f"Daily News {today_str} — " + (
        f"매수{n_buy} 매도{n_closed_today} 보유{len(hist['positions'])}" if regime_on
        else f"현금대기 (매도{n_closed_today} 보유{len(hist['positions'])})")
    send_email(subject, body)
    print(f"\n메일 발송 완료 → {RECEIVE_EMAIL}")


if __name__ == "__main__":
    main()
