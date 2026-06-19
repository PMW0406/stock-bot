"""
자동매매 봇 (GitHub Actions용)
- 아침 8시: 종목 스캔 → 매수
- 보유종목 관리: 10영업일 후 매도 / 손절 -4%
- 종목당 투자금액: 전체 예수금의 20% (최대 5종목)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PW  = os.environ.get("GMAIL_APP_PW", "")
RECEIVE_EMAIL = os.environ.get("RECEIVE_EMAIL", "")

from kis_trader import get_balance, buy_market, sell_market, get_current_price

# 설정
MAX_STOCKS    = 5      # 최대 보유 종목 수
INVEST_RATIO  = 0.20   # 종목당 예수금 비율 (20%)
STOP_LOSS     = -0.04  # 손절 -4%
HOLD_DAYS     = 10     # 보유 영업일
MIN_SCORE     = 70
MARKET_CAP_MIN = 100_000_000_000
MARKET_CAP_MAX = 5_000_000_000_000

PORTFOLIO_FILE = "/tmp/portfolio.json"  # 매수 기록


def load_portfolio():
    if Path(PORTFOLIO_FILE).exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {}


def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def get_trading_days_count(buy_date_str):
    """매수일로부터 오늘까지 영업일 수"""
    buy_date = datetime.strptime(buy_date_str, "%Y-%m-%d")
    today    = datetime.today()
    df       = fdr.DataReader("005930",
                              buy_date.strftime("%Y-%m-%d"),
                              today.strftime("%Y-%m-%d"))
    return max(0, len(df) - 1)


# ── 전략 함수 (morning_bot_cloud.py 와 동일) ──────────────────────────

def check_market_state():
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
    market = {}
    for code in ["KS11", "KQ11"]:
        name = "코스피" if code == "KS11" else "코스닥"
        try:
            df    = fdr.DataReader(code, start, end)
            close = df["Close"]
            ma5   = close.rolling(5).mean()
            ma20  = close.rolling(20).mean()
            c, m5, m20 = float(close.iloc[-1]), float(ma5.iloc[-1]), float(ma20.iloc[-1])
            if c >= m20 and m5 >= m20:
                ret5 = (c - float(close.iloc[-6])) / float(close.iloc[-6])
                market[code] = ret5 >= 0
            else:
                market[code] = False
        except:
            market[code] = False
    return market.get("KS11", False), market.get("KQ11", False)


def swing_score_and_detail(df):
    if len(df) < 65: return 0, False, {}
    ref_idx = len(df) - 1
    window  = df.iloc[max(0, ref_idx - 252): ref_idx + 1]
    today   = df.iloc[ref_idx]
    score   = 0

    close  = window["Close"]
    volume = window["Volume"]
    ma5    = close.rolling(5).mean()
    ma20   = close.rolling(20).mean()
    ma60   = close.rolling(60).mean()
    ma5_now, ma20_now, ma60_now = float(ma5.iloc[-1]), float(ma20.iloc[-1]), float(ma60.iloc[-1])

    if any(np.isnan([ma5_now, ma20_now, ma60_now])): return 0, False, {}

    if today["Close"] > ma20_now: score += 10
    if ma20_now > ma60_now:       score += 10
    ma60_prev = float(ma60.iloc[-11]) if len(ma60) > 11 else np.nan
    if not np.isnan(ma60_prev) and ma60_now > ma60_prev: score += 10

    ret_20 = (today["Close"] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) > 21 else 0
    if ret_20 > 5:   score += 20
    elif ret_20 > 0: score += 10

    recent_high = close.iloc[-8:-1].max() if len(close) >= 8 else close.max()
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

    prev_close = float(df.iloc[ref_idx - 1]["Close"])
    prev_ma5   = float(ma5.iloc[-2]) if len(ma5) >= 2 else np.nan
    if not np.isnan(prev_ma5) and (prev_close < prev_ma5) and (today["Close"] > ma5_now):
        score += 10

    # 종가 위치
    h_now    = float(df.iloc[ref_idx]["High"])
    l_now    = float(df.iloc[ref_idx]["Low"])
    hl_range = h_now - l_now
    cl       = (today["Close"] - l_now) / hl_range if hl_range > 0 else 0.5

    is_bearish_vol = (today["Close"] < today["Open"]) and (vol_ratio >= 2.0)
    must_pass = (
        today["Close"] > ma20_now and ma20_now > ma60_now and
        -8 <= pullback <= -0.5 and
        vol_ratio >= 1.0 and vol_ratio <= 2.5 and
        vol_decrease and not is_bearish_vol and
        0.40 <= cl <= 0.85
    )
    return score, must_pass, {"pullback": pullback, "vol_ratio": vol_ratio, "close_loc": cl}


def scan_candidates(kospi_ok, kosdaq_ok):
    start = (datetime.today() - timedelta(days=150)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    dfs = []
    if kospi_ok:
        k = fdr.StockListing("KOSPI"); k["market"] = "KOSPI"; dfs.append(k)
    if kosdaq_ok:
        q = fdr.StockListing("KOSDAQ"); q["market"] = "KOSDAQ"; dfs.append(q)

    all_s    = pd.concat(dfs, ignore_index=True)
    filtered = all_s[(all_s["Marcap"] >= MARKET_CAP_MIN) & (all_s["Marcap"] <= MARKET_CAP_MAX)]
    name_map = dict(zip(filtered["Code"], filtered["Name"]))
    all_ret20 = {}
    results   = []

    for ticker in filtered["Code"].tolist():
        try:
            df = fdr.DataReader(ticker, start, end)
            if df.empty or len(df) < 65: continue
            df.index = pd.to_datetime(df.index)
            if len(df) > 21:
                ret20 = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-21])) / float(df["Close"].iloc[-21]) * 100
                all_ret20[ticker] = ret20
            score, must_pass, detail = swing_score_and_detail(df)
            if not must_pass or score < MIN_SCORE: continue
            results.append({
                "종목코드": ticker,
                "종목명":   name_map.get(ticker, ticker),
                "점수":     score,
                "현재가":   float(df["Close"].iloc[-1]),
                "ret20":    all_ret20.get(ticker, 0),
            })
        except: continue

    if all_ret20:
        rs_thr  = np.percentile(list(all_ret20.values()), 80)
        results = [r for r in results if r["ret20"] >= rs_thr]

    return sorted(results, key=lambda x: x["점수"], reverse=True)


# ── 매도 관리 ────────────────────────────────────────────────────────

def manage_sells(portfolio, holdings):
    """손절 or 10영업일 도달 종목 매도"""
    sold = []
    holding_map = {h["종목코드"]: h for h in holdings}

    for ticker, info in list(portfolio.items()):
        if ticker not in holding_map: continue
        h          = holding_map[ticker]
        buy_price  = info["buy_price"]
        curr_price = h["현재가"]
        ret        = (curr_price - buy_price) / buy_price
        days_held  = get_trading_days_count(info["buy_date"])

        reason = None
        if ret <= STOP_LOSS:
            reason = f"손절 ({ret*100:.1f}%)"
        elif days_held >= HOLD_DAYS:
            reason = f"{days_held}영업일 보유"

        if reason:
            qty = h["보유수량"]
            ok, msg = sell_market(ticker, qty)
            if ok:
                sold.append(f"{info['종목명']} {qty}주 매도 [{reason}] → {curr_price:,.0f}원 (수익률 {ret*100:+.1f}%)")
                del portfolio[ticker]
            else:
                sold.append(f"{info['종목명']} 매도 실패: {msg}")

    return sold


# ── 매수 실행 ────────────────────────────────────────────────────────

def execute_buys(candidates, portfolio, cash):
    """후보 종목 매수"""
    bought = []
    current_count = len(portfolio)

    for r in candidates:
        if current_count >= MAX_STOCKS: break
        if r["종목코드"] in portfolio: continue  # 이미 보유

        invest_amt = cash * INVEST_RATIO
        price      = r["현재가"]
        qty        = int(invest_amt // price)
        if qty < 1: continue

        ok, msg = buy_market(r["종목코드"], qty)
        if ok:
            portfolio[r["종목코드"]] = {
                "종목명":   r["종목명"],
                "buy_price": price,
                "buy_date":  datetime.today().strftime("%Y-%m-%d"),
                "qty":       qty,
                "score":     r["점수"],
            }
            bought.append(f"{r['종목명']} {qty}주 매수 @ {price:,.0f}원 (점수:{r['점수']})")
            current_count += 1
            cash -= price * qty
        else:
            bought.append(f"{r['종목명']} 매수 실패: {msg}")

    return bought


# ── 메일 발송 ────────────────────────────────────────────────────────

def send_trade_email(bought, sold, holdings, cash):
    if not GMAIL_ADDRESS or not RECEIVE_EMAIL:
        return
    if not bought and not sold:
        return

    today = datetime.today().strftime("%Y-%m-%d")

    bought_rows = ""
    for b in bought:
        bought_rows += f"<tr><td style='padding:8px;color:#00c853;'>✅ 매수</td><td style='padding:8px;'>{b}</td></tr>"

    sold_rows = ""
    for s in sold:
        color = "#ff1744" if "손절" in s else "#64b5f6"
        icon  = "🛑" if "손절" in s else "✅"
        sold_rows += f"<tr><td style='padding:8px;color:{color};'>{icon} 매도</td><td style='padding:8px;'>{s}</td></tr>"

    holding_rows = ""
    for h in holdings:
        ret_color = "#00c853" if h["수익률"] >= 0 else "#ff1744"
        holding_rows += f"""
        <tr>
          <td style='padding:8px;'>{h['종목명']}</td>
          <td style='padding:8px;text-align:right;'>{h['보유수량']}주</td>
          <td style='padding:8px;text-align:right;'>{h['현재가']:,.0f}원</td>
          <td style='padding:8px;text-align:right;color:{ret_color};'>{h['수익률']:+.1f}%</td>
        </tr>"""

    html = f"""
<html><body style='background:#0d0d0d;color:#fff;font-family:sans-serif;padding:20px;'>
  <div style='max-width:600px;margin:auto;'>
    <h2 style='color:#64b5f6;'>🤖 자동매매 체결 알림 - {today}</h2>

    <table style='width:100%;background:#1e1e1e;border-radius:8px;margin-bottom:16px;border-collapse:collapse;'>
      {bought_rows}{sold_rows}
    </table>

    <div style='background:#1e1e1e;padding:12px;border-radius:8px;margin-bottom:16px;'>
      <b>예수금:</b> {cash:,.0f}원
    </div>

    {'<h3 style="color:#aaa;">보유종목 현황</h3><table style="width:100%;background:#1e1e1e;border-radius:8px;border-collapse:collapse;"><tr style="color:#aaa;font-size:13px;background:#2d2d2d;"><th style="padding:8px;text-align:left;">종목</th><th style="padding:8px;text-align:right;">수량</th><th style="padding:8px;text-align:right;">현재가</th><th style="padding:8px;text-align:right;">수익률</th></tr>' + holding_rows + '</table>' if holdings else ''}

    <p style='color:#555;font-size:12px;margin-top:16px;'>자동매매 봇 | 투자 결과는 본인 책임입니다.</p>
  </div>
</body></html>"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[자동매매] {today} 체결 알림 (매수:{len(bought)} 매도:{len(sold)})"
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = RECEIVE_EMAIL
        msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            smtp.sendmail(GMAIL_ADDRESS, RECEIVE_EMAIL, msg.as_string())
        print(f"체결 알림 메일 발송 → {RECEIVE_EMAIL}")
    except Exception as e:
        print(f"메일 발송 실패: {e}")


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*45}")
    print(f"자동매매 봇 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*45}\n")

    # 1. 잔고 조회
    holdings, cash = get_balance()
    print(f"예수금: {cash:,.0f}원 | 보유종목: {len(holdings)}개")

    # 2. 포트폴리오 로드
    portfolio = load_portfolio()
    print(f"추적 중인 종목: {len(portfolio)}개")

    # 3. 매도 관리 (손절/기간 만료)
    sold = []
    if holdings:
        print("\n[매도 검토]")
        sold = manage_sells(portfolio, holdings)
        for s in sold: print(f"  {s}")
        if not sold: print("  해당 없음")
        save_portfolio(portfolio)

    # 4. 시장 필터
    kospi_ok, kosdaq_ok = check_market_state()
    print(f"\n[시장 상태] 코스피: {'양호' if kospi_ok else '이탈'} / 코스닥: {'양호' if kosdaq_ok else '이탈'}")

    if not kospi_ok and not kosdaq_ok:
        print("시장 조건 미충족 → 매수 없음")
        return

    # 5. 현재 보유 종목 수 확인
    if len(portfolio) >= MAX_STOCKS:
        print(f"\n최대 보유 종목 수 도달 ({MAX_STOCKS}개) → 매수 없음")
        return

    # 6. 종목 스캔
    print("\n[종목 스캔 중...]")
    candidates = scan_candidates(kospi_ok, kosdaq_ok)
    print(f"후보: {len(candidates)}개")

    if not candidates:
        print("조건 통과 종목 없음")
        return

    # 7. 매수 실행
    print("\n[매수 실행]")
    bought = execute_buys(candidates[:MAX_STOCKS], portfolio, cash)
    for b in bought: print(f"  {b}")
    save_portfolio(portfolio)

    # 8. 체결 알림 메일
    holdings, cash = get_balance()
    send_trade_email(bought, sold, holdings, cash)

    print(f"\n완료! 포트폴리오: {len(portfolio)}개 종목")


if __name__ == "__main__":
    main()
