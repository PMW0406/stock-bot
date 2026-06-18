"""
매일 아침 8시 자동 실행 주식 봇 (GitHub Actions용)
설정값을 환경변수에서 읽음 (config.py 대신)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import time, warnings
warnings.filterwarnings("ignore")

# 환경변수에서 설정 읽기
GMAIL_ADDRESS     = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PW      = os.environ["GMAIL_APP_PW"]
RECEIVE_EMAIL     = os.environ["RECEIVE_EMAIL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

MIN_SCORE      = 8
TOP_N          = 5
TARGET_GAIN    = 0.05
STOP_LOSS      = 0.03
US_THRESHOLD   = 0.005
MARKET_CAP_MIN = 100_000_000_000
MARKET_CAP_MAX = 5_000_000_000_000


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df, period=14):
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def check_us_market():
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    sp  = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze().pct_change()
    nq  = yf.download("^IXIC", start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze().pct_change()
    sox = yf.download("^SOX",  start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze().pct_change()

    sp_ret  = float(sp.iloc[-1])
    nq_ret  = float(nq.iloc[-1])
    sox_ret = float(sox.iloc[-1])
    us_date = sp.index[-1].strftime("%Y-%m-%d")

    is_hojae = sp_ret >= US_THRESHOLD or nq_ret >= US_THRESHOLD
    return is_hojae, sp_ret, nq_ret, sox_ret, us_date


def tech_score_and_detail(df):
    if len(df) < 21:
        return 0, {}

    ref_idx = len(df) - 1
    window  = df.iloc[max(0, ref_idx - 252): ref_idx + 1]
    today   = df.iloc[ref_idx]
    score   = 0
    detail  = {}

    vol_avg   = window["Volume"].iloc[-20:].mean()
    vol_ratio = today["Volume"] / vol_avg if vol_avg > 0 else 0
    vs = 4 if vol_ratio>=5 else 3 if vol_ratio>=3 else 2 if vol_ratio>=2 else 1 if vol_ratio>=1.5 else 0
    score += vs
    detail["거래량"] = f"{vol_ratio:.1f}배"

    rsi_s = calc_rsi(window["Close"])
    rsi   = float(rsi_s.iloc[-1]) if len(rsi_s) >= 14 else 50
    rs = 3 if rsi<30 else 2 if rsi<35 else 1 if rsi<40 else 0
    score += rs
    detail["RSI"] = f"{rsi:.0f}"

    if ref_idx >= 1:
        drop = (today["Close"] - df.iloc[ref_idx-1]["Close"]) / df.iloc[ref_idx-1]["Close"] * 100
        ds = 2 if -10<=drop<=-7 else 1 if -7<drop<=-3 else 0
        score += ds
        detail["하락폭"] = f"{drop:.1f}%"

    ma60 = window["Close"].rolling(60).mean().iloc[-1]
    if not np.isnan(ma60):
        ms = 2 if today["Close"]>ma60 else 1 if today["Close"]>ma60*0.97 else 0
        score += ms
        detail["MA60"] = "위" if today["Close"] > ma60 else "근접"

    ma5 = window["Close"].rolling(5).mean()
    if ref_idx >= 1 and len(ma5) >= 2:
        prev_close = df.iloc[ref_idx-1]["Close"]
        prev_ma5   = ma5.iloc[-2]
        if not np.isnan(prev_ma5) and prev_close < prev_ma5:
            score += 1
            if today["Open"] > prev_close:
                score += 1
            detail["5일선"] = "이탈반등"

    return score, detail


def get_candidates():
    print("종목 스캔 중...")
    start = (datetime.today() - timedelta(days=120)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    all_s    = pd.concat([fdr.StockListing("KOSPI"), fdr.StockListing("KOSDAQ")], ignore_index=True)
    filtered = all_s[(all_s["Marcap"] >= MARKET_CAP_MIN) & (all_s["Marcap"] <= MARKET_CAP_MAX)]
    name_map = dict(zip(filtered["Code"], filtered["Name"]))
    tickers  = filtered["Code"].tolist()

    results = []
    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, start, end)
            if df.empty or len(df) < 21:
                continue
            score, detail = tech_score_and_detail(df)
            if score < MIN_SCORE:
                continue

            latest    = df.iloc[-1]
            buy_price = float(latest["Open"])
            ma20      = df["Close"].rolling(20).mean().iloc[-1]
            target_pct = min(15, max(5, (ma20 - buy_price) / buy_price * 100))
            target    = buy_price * (1 + target_pct / 100)
            stop      = buy_price * (1 - STOP_LOSS)

            results.append({
                "종목코드":  ticker,
                "종목명":    name_map.get(ticker, ticker),
                "점수":      score,
                "시가":      buy_price,
                "목표가":    target,
                "목표수익":  target_pct,
                "손절가":    stop,
                "detail":    detail,
            })
        except:
            continue

    results = sorted(results, key=lambda x: x["점수"], reverse=True)[:TOP_N]
    print(f"   -> {len(results)}개 후보 선정")
    return results


def analyze_with_claude(candidates, sp_ret, nq_ret, sox_ret):
    if not ANTHROPIC_API_KEY:
        return None, {}

    try:
        import anthropic, re
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        stocks_info = ""
        for r in candidates:
            d = r["detail"]
            stocks_info += f"""
종목: {r['종목명']} (점수:{r['점수']}점)
- 거래량: {d.get('거래량','N/A')} / RSI: {d.get('RSI','N/A')} / 하락폭: {d.get('하락폭','N/A')}
- MA60: {d.get('MA60','N/A')} / 5일선: {d.get('5일선','없음')}
- 시가: {r['시가']:,.0f}원
"""

        prompt = f"""
미국 시장: S&P500 {sp_ret*100:+.2f}% / 나스닥 {nq_ret*100:+.2f}% / SOX {sox_ret*100:+.2f}%

아래 한국 주식 종목들의 기술적 지표를 보고 각 종목마다:
1. 단기 목표가 (% 기준, 근거 포함)
2. 손절가 (% 기준, 근거 포함)
을 한 줄씩 간결하게 분석해주세요.

{stocks_info}

반드시 아래 형식을 지켜주세요 (숫자는 양수로만):
[종목명]
목표: +X% (근거)
손절: -X% (근거)
"""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text

        prices = {}
        current = None
        for line in text.splitlines():
            m = re.match(r"\[(.+)\]", line.strip())
            if m:
                current = m.group(1).strip()
                prices[current] = {}
            if current:
                t = re.search(r"목표\s*:\s*\+?(\d+\.?\d*)\s*%", line)
                s = re.search(r"손절\s*:\s*-?(\d+\.?\d*)\s*%", line)
                if t:
                    prices[current]["target_pct"] = float(t.group(1))
                if s:
                    prices[current]["stop_pct"] = float(s.group(1))

        return text, prices
    except Exception as e:
        print(f"Claude API 오류: {e}")
        return None, {}


def send_email(subject, body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = RECEIVE_EMAIL
    msg.attach(MIMEText(body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PW)
        smtp.sendmail(GMAIL_ADDRESS, RECEIVE_EMAIL, msg.as_string())


def build_email(candidates, sp_ret, nq_ret, sox_ret, us_date, claude_analysis, claude_prices=None):
    today = datetime.today().strftime("%Y-%m-%d")
    if claude_prices is None:
        claude_prices = {}

    rows = ""
    for i, r in enumerate(candidates, 1):
        d = r["detail"]
        detail_str = " / ".join([f"{k}: {v}" for k, v in d.items()])

        cp = claude_prices.get(r["종목명"], {})
        target_pct = cp.get("target_pct", r["목표수익"])
        stop_pct   = cp.get("stop_pct",   STOP_LOSS * 100)
        target_val = r["시가"] * (1 + target_pct / 100)
        stop_val   = r["시가"] * (1 - stop_pct   / 100)
        ai_tag     = " <span style='color:#7c4dff;font-size:11px;'>AI</span>" if cp else ""

        rows += f"""
        <tr>
          <td style="padding:10px;font-weight:bold;font-size:16px;">{i}. {r['종목명']}</td>
          <td style="padding:10px;">{r['점수']}점</td>
          <td style="padding:10px;">{r['시가']:,.0f}원</td>
          <td style="padding:10px;color:#00c853;">+{target_pct:.1f}% ({target_val:,.0f}원){ai_tag}</td>
          <td style="padding:10px;color:#ff1744;">-{stop_pct:.1f}% ({stop_val:,.0f}원){ai_tag}</td>
        </tr>
        <tr>
          <td colspan="5" style="padding:4px 10px 12px;color:#888;font-size:13px;">{detail_str}</td>
        </tr>
"""

    import html as html_module
    claude_section = ""
    if claude_analysis:
        claude_section = f"""
        <div style="background:#1a1a2e;padding:16px;border-radius:8px;margin-top:20px;">
          <h3 style="color:#7c4dff;margin:0 0 10px;">Claude AI 분석 근거</h3>
          <pre style="color:#ccc;font-size:13px;white-space:pre-wrap;">{html_module.escape(claude_analysis)}</pre>
        </div>
"""

    html = f"""
<html><body style="background:#0d0d0d;color:#fff;font-family:sans-serif;padding:20px;">
  <div style="max-width:600px;margin:auto;">
    <h2 style="color:#64b5f6;">📈 주식 매수 신호 - {today}</h2>

    <div style="background:#1e1e1e;padding:14px;border-radius:8px;margin-bottom:20px;">
      <b>미국 시장 ({us_date})</b><br>
      S&P500: <span style="color:#{'00c853' if sp_ret>0 else 'ff1744'}">{sp_ret*100:+.2f}%</span> &nbsp;
      나스닥: <span style="color:#{'00c853' if nq_ret>0 else 'ff1744'}">{nq_ret*100:+.2f}%</span> &nbsp;
      SOX: <span style="color:#{'00c853' if sox_ret>0 else 'ff1744'}">{sox_ret*100:+.2f}%</span>
    </div>

    <table style="width:100%;border-collapse:collapse;background:#1e1e1e;border-radius:8px;">
      <tr style="background:#2d2d2d;color:#aaa;font-size:13px;">
        <th style="padding:10px;text-align:left;">종목</th>
        <th style="padding:10px;">점수</th>
        <th style="padding:10px;">시가</th>
        <th style="padding:10px;color:#00c853;">목표가</th>
        <th style="padding:10px;color:#ff1744;">손절가</th>
      </tr>
      {rows}
    </table>

    {claude_section}

    <p style="color:#555;font-size:12px;margin-top:20px;">
      목표가/손절가는 Claude AI 분석 기반 (보라색 AI 표시)<br>
      승률 72% / 평균 수익 11.6% (3개월 백테스트 기준)<br>
      투자는 본인 판단 하에 진행하세요.
    </p>
  </div>
</body></html>
"""
    return html


def main():
    print(f"\n{'='*40}")
    print(f"주식 봇 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*40}\n")

    is_hojae, sp_ret, nq_ret, sox_ret, us_date = check_us_market()
    print(f"미국 시장 ({us_date}): S&P500 {sp_ret*100:+.2f}% / 나스닥 {nq_ret*100:+.2f}%")

    if not is_hojae:
        print("호재 없음 → 신호 없음 메일 발송")
        send_email(
            f"[주식봇] {datetime.today().strftime('%Y-%m-%d')} 오늘 신호 없음",
            f"<html><body style='background:#0d0d0d;color:#fff;padding:20px;'>"
            f"<h3>오늘 미국 시장 호재 없음</h3>"
            f"<p>S&P500: {sp_ret*100:+.2f}% / 나스닥: {nq_ret*100:+.2f}%</p>"
            f"<p>매수 신호 없습니다.</p></body></html>"
        )
        return

    candidates = get_candidates()
    if not candidates:
        print("조건 통과 종목 없음")
        return

    claude_analysis, claude_prices = analyze_with_claude(candidates, sp_ret, nq_ret, sox_ret)

    body    = build_email(candidates, sp_ret, nq_ret, sox_ret, us_date, claude_analysis, claude_prices)
    subject = f"[주식봇] {datetime.today().strftime('%Y-%m-%d')} 매수 후보 {len(candidates)}종목"
    send_email(subject, body)

    print(f"\n이메일 발송 완료 → {RECEIVE_EMAIL}")
    for r in candidates:
        print(f"  {r['종목명']} | {r['점수']}점 | 시가 {r['시가']:,.0f}원")


if __name__ == "__main__":
    main()
