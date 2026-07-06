# -*- coding: utf-8 -*-
"""
주식봇 v14 — 52주 신고가 스윙 (매일 아침 7:10 KST, GitHub Actions)

전략 (5년 백테스트 전 연도 플러스 검증):
  국면: 코스피 종가 > 120일선  → 아니면 전량 현금 대기
  진입: 시총 1000억~5조 + 20일평균 거래대금 30억↑
        + 52주 최고가 대비 -5% 이내 + MA5 > MA20
        (시가 갭 +2% 이상이면 다음날 자동 진입취소)
  청산: 15거래일 보유 or -8% 손절 (조기 익절 없음)
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
STOP_LOSS         = 0.08                # 손절 -8%
NEAR_HIGH         = -5.0                # 52주 고가 대비 -5% 이내
GAP_MAX           = 2.0                 # 시가 갭 +2% 이상이면 진입취소
GAP_MIN           = -3.0                # 시가 갭 -3% 이하(급락 출발)도 진입취소
BREAKER_WINDOW    = 10                  # 서킷브레이커: 최근 청산 10건 중
BREAKER_STOPS     = 7                   #   손절이 7건 이상이면
BREAKER_PAUSE     = 5                   #   5거래일 신규진입 중단
MARKET_CAP_MIN    = 100_000_000_000
MARKET_CAP_MAX    = 5_000_000_000_000
MIN_TRADING_VALUE = 3_000_000_000       # 거래대금 30억
REGIME_MA         = 120                 # 코스피 국면 이평
HISTORY_PATH      = "history.json"


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
    """국면: 코스피 종가 > 120일선"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=300)).strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader("KS11", start, end)
        close = df["Close"]
        if len(close) < REGIME_MA + 5:
            return False, "코스피 데이터 부족"
        c    = float(close.iloc[-1])
        ma   = float(close.rolling(REGIME_MA).mean().iloc[-1])
        dist = (c / ma - 1) * 100
        if c > ma:
            return True, f"코스피 {c:,.0f} > 120일선 {ma:,.0f} ({dist:+.1f}%) — 매매 ON"
        return False, f"코스피 {c:,.0f} < 120일선 {ma:,.0f} ({dist:+.1f}%) — 현금 대기"
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

            # 1) pending 체결 확정 (진입일 시가 = 체결가, 갭 +2%↑ 또는 -3%↓면 취소)
            if p.get("entry_price") is None:
                if p["entry_date"] in df.index:
                    o = float(df.loc[p["entry_date"], "Open"])
                    gap = (o - p["ref_close"]) / p["ref_close"] * 100 if p.get("ref_close") else 0
                    if gap >= GAP_MAX or gap <= GAP_MIN:
                        cancels.append({**p, "reason": f"갭 {gap:+.1f}% 진입취소"})
                        continue
                    p["entry_price"] = round(o, 2)
                    p["stop_price"]  = round(o * (1 - STOP_LOSS), 2)
                else:
                    kept.append(p); continue    # 아직 진입일 시세 없음 (휴장 등)

            held = df[df.index >= p["entry_date"]]
            if held.empty:
                kept.append(p); continue
            days_held = len(held)
            cur = float(held["Close"].iloc[-1])
            ret = (cur - p["entry_price"]) / p["entry_price"] * 100

            # 2) 손절: 보유 중 저가가 손절가 이탈
            low_min = float(held["Low"].min())
            if low_min <= p["stop_price"]:
                breach_date = held.index[held["Low"] <= p["stop_price"]][0]
                hist["closed"].append({
                    "code": p["code"], "name": p["name"],
                    "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                    "exit_date": breach_date, "exit_price": p["stop_price"],
                    "ret_pct": round(-STOP_LOSS * 100, 2), "reason": "손절 -8%",
                })
                sell_alerts.append({"name": p["name"], "code": p["code"],
                                    "msg": f"🛑 손절가 {p['stop_price']:,.0f}원 이탈 ({breach_date}) — 아직 보유중이면 매도"})
                continue

            # 3) 만기: 15거래일 도래
            if days_held >= HOLD_DAYS:
                hist["closed"].append({
                    "code": p["code"], "name": p["name"],
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


# ─────────────────────────────────────────
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
            if d52 < NEAR_HIGH: continue                      # 신고가 -5% 이내
            avg_value = float((vol * close).rolling(20).mean().iloc[-1])
            if np.isnan(avg_value) or avg_value < MIN_TRADING_VALUE: continue
            ma5  = float(close.rolling(5).mean().iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            if not (ma5 > ma20): continue                     # 단기추세 보강 필터
            results.append({
                "code": tk, "name": name_map.get(tk, tk),
                "close": c, "hi52": h52, "d52": round(d52, 2),
                "avg_value_억": round(avg_value / 100_000_000, 1),
            })
        except:
            continue

    results.sort(key=lambda x: -x["d52"])   # 신고가 최근접순
    print(f"   -> 후보 {len(results)}개")
    return results


# ─────────────────────────────────────────
def build_email(regime_on, regime_msg, sp_ret, nq_ret, sox_ret, us_date,
                positions, sell_alerts, cancels, candidates, new_entries):
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

    # 보유 포지션
    pos_rows = ""
    for p in sorted(positions, key=lambda x: x.get("days_held", 0), reverse=True):
        pending = p.get("entry_price") is None
        ret = p.get("ret_pct"); col = g if (ret or 0) >= 0 else r
        pos_rows += f"""<tr>
          <td style="padding:8px;">{p['name']}<span style="color:#666;font-size:11px;"> {p['code']}</span></td>
          <td style="padding:8px;text-align:center;">{p['entry_date'][5:]}</td>
          <td style="padding:8px;text-align:right;">{'체결대기' if pending else f"{p['entry_price']:,.0f}"}</td>
          <td style="padding:8px;text-align:right;">{f"{p.get('current',0):,.0f}" if not pending else '-'}</td>
          <td style="padding:8px;text-align:right;color:{col};font-weight:bold;">{f"{ret:+.1f}%" if ret is not None else '-'}</td>
          <td style="padding:8px;text-align:center;">{p.get('days_held','-')}/{HOLD_DAYS}일</td>
          <td style="padding:8px;text-align:right;color:{r};">{f"{p.get('stop_price',0):,.0f}" if not pending else '-'}</td>
        </tr>"""
    pos_html = f"""<div style="background:#1e1e1e;padding:14px;border-radius:8px;margin-bottom:14px;">
      <h3 style="color:#64b5f6;margin:0 0 10px;">💼 보유 {len(positions)}/{SLOTS}</h3>
      <table style="width:100%;border-collapse:collapse;font-size:13px;color:#ddd;">
        <tr style="color:#888;font-size:12px;"><th style="text-align:left;padding:8px;">종목</th><th>진입일</th>
        <th style="text-align:right;">진입가</th><th style="text-align:right;">현재가</th>
        <th style="text-align:right;">수익률</th><th>보유일</th><th style="text-align:right;">손절가</th></tr>
        {pos_rows if pos_rows else '<tr><td colspan="7" style="padding:12px;color:#666;">보유 없음</td></tr>'}
      </table></div>"""

    # 신규 매수
    buy_html = ""
    if new_entries:
        rows = "".join(f"""<tr>
          <td style="padding:8px;font-weight:bold;">{i}. {c['name']}<span style="color:#666;font-size:11px;"> {c['code']}</span></td>
          <td style="padding:8px;text-align:right;">{c['close']:,.0f}원</td>
          <td style="padding:8px;text-align:right;color:{g};">신고가 {c['d52']:+.1f}%</td>
          <td style="padding:8px;text-align:right;">{c['avg_value_억']:,.0f}억</td>
        </tr>""" for i, c in enumerate(new_entries, 1))
        buy_html = f"""<div style="background:#1a2a1a;border:1px solid #00c853;padding:14px;border-radius:8px;margin-bottom:14px;">
          <h3 style="color:{g};margin:0 0 8px;">🎯 오늘 매수 (시가) — {len(new_entries)}종목</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px;color:#ddd;">{rows}</table>
          <p style="color:#888;font-size:12px;margin:8px 0 0;">※ 시가가 전일종가 대비 +{GAP_MAX:.0f}% 이상 갭상승이면 매수 보류 (봇도 자동 취소 처리)</p></div>"""
    elif regime_on:
        buy_html = """<div style="background:#1e1e1e;padding:12px 14px;border-radius:8px;margin-bottom:14px;color:#888;font-size:13px;">
          오늘 신규 매수 없음 (슬롯 가득 또는 후보 없음)</div>"""

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
      전략 v14: 코스피>120일선 국면 + 52주 신고가 -5%이내 + 거래대금30억 + MA5>MA20 |
      청산: {HOLD_DAYS}거래일 or -8% 손절 | 분산 {SLOTS}슬롯<br>
      5년 백테스트(2022~2026) 전 연도 플러스 · CAGR 10~18% · MDD -19~-31% (완전분산 기준, 실전은 이보다 낮게 기대)<br>
      투자는 본인 판단 하에 진행하세요.
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

    # 3) 신규 후보 스캔 (국면 ON + 브레이커 미발동일 때만)
    candidates, new_entries = [], []
    if regime_on and breaker_active:
        print("서킷브레이커 활성 → 스캔 생략 (보유종목 관리만)")
    elif regime_on:
        held_codes = {p["code"] for p in hist["positions"]}
        candidates = get_candidates(held_codes)
        empty = SLOTS - len(hist["positions"])
        for c in candidates[:max(0, empty)]:
            ref_close = c["close"]     # 전일 종가 (오늘 시가로 체결, 내일 확정)
            hist["positions"].append({
                "code": c["code"], "name": c["name"],
                "entry_date": today_str, "entry_price": None,
                "ref_close": ref_close, "stop_price": None,
            })
            new_entries.append(c)
        print(f"신규 매수 {len(new_entries)}개 (빈슬롯 {empty})")
    else:
        print("약세장 → 스캔 생략, 현금 대기")

    save_history(hist)

    # 3) candidates.json (Streamlit 앱용)
    with open("candidates.json", "w", encoding="utf-8") as f:
        json.dump({
            "format": "v14",
            "updated": datetime.today().strftime("%Y-%m-%d %H:%M"),
            "regime_on": regime_on, "regime_msg": regime_msg,
            "candidates": candidates[:20],
            "new_entries": [c["code"] for c in new_entries],
        }, f, ensure_ascii=False, indent=2)

    # 4) 메일
    body = build_email(regime_on, regime_msg, sp_ret, nq_ret, sox_ret, us_date,
                       hist["positions"], sell_alerts, cancels, candidates, new_entries)
    n_closed_today = len(sell_alerts)
    subject = f"Daily News {today_str} — " + (
        f"매수{len(new_entries)} 매도{n_closed_today} 보유{len(hist['positions'])}" if regime_on
        else f"현금대기 (매도{n_closed_today} 보유{len(hist['positions'])})")
    send_email(subject, body)
    print(f"\n메일 발송 완료 → {RECEIVE_EMAIL}")


if __name__ == "__main__":
    main()
