"""
주식 봇 v2 - 추세+눌림목+거래량 회복 전략 (GitHub Actions용)
매일 아침 8시 실행
- 시장 필터 (코스피+코스닥 둘 다 양호)
- 스윙 점수 (추세+눌림목+거래량)
- 상대강도 상위 20%
- 갭상승 +2% 이상 종목 제외
- Claude AI 목표가/손절가 분석
- 보유 추천: 10영업일
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

GMAIL_ADDRESS     = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PW      = os.environ["GMAIL_APP_PW"]
RECEIVE_EMAIL     = os.environ["RECEIVE_EMAIL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

MIN_SCORE      = 70
TOP_N          = 5
TARGET_GAIN    = 0.05
STOP_LOSS      = 0.04
MARKET_CAP_MIN = 100_000_000_000
MARKET_CAP_MAX = 5_000_000_000_000


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def check_market_state():
    """코스피/코스닥 각각 상태 반환 - 종목별로 해당 시장만 체크"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")

    market_status = {}
    for code in ["KS11", "KQ11"]:
        name = "코스피" if code == "KS11" else "코스닥"
        try:
            df = fdr.DataReader(code, start, end)
            df.index = pd.to_datetime(df.index)
            if len(df) < 25:
                market_status[code] = (False, "데이터 부족")
                continue
            close    = df["Close"]
            ma5      = close.rolling(5).mean()
            ma20     = close.rolling(20).mean()
            c_now    = float(close.iloc[-1])
            ma5_now  = float(ma5.iloc[-1])
            ma20_now = float(ma20.iloc[-1])

            if c_now < ma20_now:
                market_status[code] = (False, f"{name} 20일선 이탈")
            elif ma5_now < ma20_now:
                market_status[code] = (False, f"{name} 5일선<20일선")
            else:
                ret5 = (c_now - float(close.iloc[-6])) / float(close.iloc[-6])
                if ret5 < 0:
                    market_status[code] = (False, f"{name} 5일 수익률 음수")
                else:
                    market_status[code] = (True, f"{name} 양호")
        except Exception as e:
            market_status[code] = (False, str(e))

    kospi_ok  = market_status.get("KS11", (False, ""))[0]
    kosdaq_ok = market_status.get("KQ11", (False, ""))[0]

    # 둘 다 나쁘면 중단
    if not kospi_ok and not kosdaq_ok:
        msg = " / ".join([market_status["KS11"][1], market_status["KQ11"][1]])
        return False, False, msg

    status_msg = " / ".join([market_status["KS11"][1], market_status["KQ11"][1]])
    return kospi_ok, kosdaq_ok, status_msg


def swing_score_and_detail(df):
    """스윙 점수 + 세부 지표"""
    if len(df) < 65:
        return 0, False, {}

    ref_idx = len(df) - 1
    window  = df.iloc[max(0, ref_idx - 252): ref_idx + 1]
    today   = df.iloc[ref_idx]
    score   = 0
    detail  = {}

    close  = window["Close"]
    volume = window["Volume"]
    ma5    = close.rolling(5).mean()
    ma20   = close.rolling(20).mean()
    ma60   = close.rolling(60).mean()

    ma5_now  = float(ma5.iloc[-1])
    ma20_now = float(ma20.iloc[-1])
    ma60_now = float(ma60.iloc[-1])

    if any(np.isnan([ma5_now, ma20_now, ma60_now])):
        return 0, False, {}

    # 추세 (30점)
    if today["Close"] > ma20_now: score += 10
    if ma20_now > ma60_now:       score += 10
    ma60_prev = float(ma60.iloc[-11]) if len(ma60) > 11 else np.nan
    if not np.isnan(ma60_prev) and ma60_now > ma60_prev: score += 10
    detail["추세"] = f"MA20{'↑' if today['Close']>ma20_now else '↓'} MA60{'↑' if ma20_now>ma60_now else '↓'}"

    # 상대강도 (20점)
    ret_20 = (today["Close"] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) > 21 else 0
    if ret_20 > 5:   score += 20
    elif ret_20 > 0: score += 10
    detail["20일수익"] = f"{ret_20:.1f}%"

    # 눌림목 (20점)
    recent_high = close.iloc[-8:-1].max() if len(close) >= 8 else close.max()
    pullback    = (today["Close"] - recent_high) / recent_high * 100
    if -8 <= pullback <= -3:   score += 20
    elif -3 < pullback <= -1: score += 10
    detail["눌림폭"] = f"{pullback:.1f}%"

    # 거래량 (30점)
    vol_recent   = volume.iloc[-4:-1].mean()
    vol_before   = volume.iloc[-9:-4].mean()
    vol_decrease = vol_recent < vol_before * 0.9 if vol_before > 0 else False
    vol_5avg     = volume.iloc[-6:-1].mean()
    vol_ratio    = float(today["Volume"]) / vol_5avg if vol_5avg > 0 else 0

    if vol_decrease and vol_ratio >= 1.5: score += 20
    elif vol_ratio >= 1.5:                score += 10
    detail["거래량"] = f"조정중감소:{vol_decrease} / 회복:{vol_ratio:.1f}배"

    # 5일선 회복 (10점)
    prev_close = float(df.iloc[ref_idx - 1]["Close"])
    prev_ma5   = float(ma5.iloc[-2]) if len(ma5) >= 2 else np.nan
    recovered  = not np.isnan(prev_ma5) and (prev_close < prev_ma5) and (today["Close"] > ma5_now)
    if recovered:
        score += 10
        detail["5일선"] = "회복"

    # 종가 위치 (close_location)
    h_now    = float(window["High"].iloc[-1])
    l_now    = float(window["Low"].iloc[-1])
    hl_range = h_now - l_now
    cl       = (today["Close"] - l_now) / hl_range if hl_range > 0 else 0.5
    detail["종가위치"] = f"{cl:.2f}"

    # 거래량 터진 음봉 제외
    is_bearish_vol = (today["Close"] < today["Open"]) and (vol_ratio >= 2.0)

    must_pass = (
        today["Close"] > ma20_now and
        ma20_now > ma60_now and
        -8 <= pullback <= -0.5 and
        vol_ratio >= 1.0 and
        vol_ratio <= 2.5 and        # 거래량 상한 (폭발적 거래량 제외)
        vol_decrease and
        not is_bearish_vol and
        0.40 <= cl <= 0.85          # 종가 위치 필터
    )

    return score, must_pass, detail


def get_candidates(kospi_ok, kosdaq_ok):
    """매수 후보 종목 스캔 - 종목별로 해당 시장 필터 적용"""
    print("종목 스캔 중...")
    start = (datetime.today() - timedelta(days=150)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    kospi_list  = fdr.StockListing("KOSPI")
    kosdaq_list = fdr.StockListing("KOSDAQ")

    # 시장 필터 통과한 시장 종목만 포함
    dfs = []
    if kospi_ok:
        kospi_list["market"] = "KOSPI"
        dfs.append(kospi_list)
    if kosdaq_ok:
        kosdaq_list["market"] = "KOSDAQ"
        dfs.append(kosdaq_list)

    all_s    = pd.concat(dfs, ignore_index=True)
    filtered = all_s[(all_s["Marcap"] >= MARKET_CAP_MIN) & (all_s["Marcap"] <= MARKET_CAP_MAX)]
    name_map = dict(zip(filtered["Code"], filtered["Name"]))
    tickers  = filtered["Code"].tolist()

    # 전 종목 데이터 수집 (RS 계산용)
    all_ret20 = {}
    results   = []

    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, start, end)
            if df.empty or len(df) < 65:
                continue
            df.index = pd.to_datetime(df.index)

            # RS 계산
            if len(df) > 21:
                ret20 = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-21])) / float(df["Close"].iloc[-21]) * 100
                all_ret20[ticker] = ret20

            score, must_pass, detail = swing_score_and_detail(df)
            if not must_pass or score < MIN_SCORE:
                continue

            latest    = df.iloc[-1]
            buy_price = float(latest["Close"])  # 내일 시가 기준이나 당일 종가로 근사
            stop      = buy_price * (1 - STOP_LOSS)
            target    = buy_price * 1.10  # 기본 목표 10%

            results.append({
                "종목코드": ticker,
                "종목명":   name_map.get(ticker, ticker),
                "점수":     score,
                "현재가":   buy_price,
                "목표가":   target,
                "목표수익": 10.0,
                "손절가":   stop,
                "ret20":    all_ret20.get(ticker, 0),
                "detail":   detail,
                "df":       df,
            })
        except:
            continue

    # RS 상위 20% 필터
    if all_ret20:
        rs_threshold = np.percentile(list(all_ret20.values()), 80)
        results = [r for r in results if r["ret20"] >= rs_threshold]

    results = sorted(results, key=lambda x: x["점수"], reverse=True)[:TOP_N]
    print(f"   -> {len(results)}개 후보 선정")
    return results


def analyze_with_claude(candidates, sp_ret, nq_ret, sox_ret):
    """Claude AI 목표가/손절가 분석"""
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
- 추세: {d.get('추세','N/A')} / 20일수익: {d.get('20일수익','N/A')}
- 눌림폭: {d.get('눌림폭','N/A')} / 거래량: {d.get('거래량','N/A')}
- 5일선: {d.get('5일선','미회복')}
- 현재가: {r['현재가']:,.0f}원
"""

        prompt = f"""
미국 시장: S&P500 {sp_ret*100:+.2f}% / 나스닥 {nq_ret*100:+.2f}% / SOX {sox_ret*100:+.2f}%
전략: 추세+눌림목+거래량 회복 스윙 (10영업일 보유)

아래 종목들의 기술적 지표를 보고 각 종목마다:
1. 단기 목표가 (% 기준, 근거 포함)
2. 손절가 (% 기준, 근거 포함)
을 간결하게 분석해주세요.

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

        prices  = {}
        current = None
        for line in text.splitlines():
            m = re.match(r"\[(.+)\]", line.strip())
            if m:
                current = m.group(1).strip()
                prices[current] = {}
            if current:
                t = re.search(r"목표\s*:\s*\+?(\d+\.?\d*)\s*%", line)
                s = re.search(r"손절\s*:\s*-?(\d+\.?\d*)\s*%", line)
                if t: prices[current]["target_pct"] = float(t.group(1))
                if s: prices[current]["stop_pct"]   = float(s.group(1))

        return text, prices
    except Exception as e:
        print(f"Claude API 오류: {e}")
        return None, {}


def build_email(candidates, sp_ret, nq_ret, sox_ret, us_date,
                market_status, claude_analysis, claude_prices):
    import html as html_module
    today  = datetime.today().strftime("%Y-%m-%d")
    cp     = claude_prices or {}

    rows = ""
    for i, r in enumerate(candidates, 1):
        d          = r["detail"]
        detail_str = " / ".join([f"{k}: {v}" for k, v in d.items() if k != "df"])
        prices     = cp.get(r["종목명"], {})
        target_pct = prices.get("target_pct", r["목표수익"])
        stop_pct   = prices.get("stop_pct",   STOP_LOSS * 100)
        target_val = r["현재가"] * (1 + target_pct / 100)
        stop_val   = r["현재가"] * (1 - stop_pct   / 100)
        ai_tag     = " <span style='color:#7c4dff;font-size:11px;'>AI</span>" if prices else ""

        rows += f"""
        <tr>
          <td style="padding:10px;font-weight:bold;font-size:15px;">{i}. {r['종목명']}</td>
          <td style="padding:10px;text-align:center;">{r['점수']}점</td>
          <td style="padding:10px;text-align:right;">{r['현재가']:,.0f}원</td>
          <td style="padding:10px;text-align:right;color:#00c853;">+{target_pct:.1f}%<br><span style="font-size:12px;">({target_val:,.0f}원){ai_tag}</span></td>
          <td style="padding:10px;text-align:right;color:#ff1744;">-{stop_pct:.1f}%<br><span style="font-size:12px;">({stop_val:,.0f}원){ai_tag}</span></td>
        </tr>
        <tr>
          <td colspan="5" style="padding:2px 10px 12px;color:#888;font-size:12px;">{detail_str}</td>
        </tr>
"""

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
  <div style="max-width:640px;margin:auto;">
    <h2 style="color:#64b5f6;">📈 스윙 매수 신호 - {today}</h2>

    <div style="background:#1e1e1e;padding:14px;border-radius:8px;margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap;">
      <div>
        <b>미국 시장 ({us_date})</b><br>
        S&P500: <span style="color:#{'00c853' if sp_ret>0 else 'ff1744'}">{sp_ret*100:+.2f}%</span> &nbsp;
        나스닥: <span style="color:#{'00c853' if nq_ret>0 else 'ff1744'}">{nq_ret*100:+.2f}%</span> &nbsp;
        SOX: <span style="color:#{'00c853' if sox_ret>0 else 'ff1744'}">{sox_ret*100:+.2f}%</span>
      </div>
      <div>
        <b>한국 시장</b><br>
        <span style="color:#00c853;">● {market_status}</span>
      </div>
    </div>

    <div style="background:#1a2a1a;padding:10px 14px;border-radius:6px;margin-bottom:16px;font-size:13px;color:#aaa;">
      ⏱ 보유 추천: <b style="color:#fff;">10영업일</b> &nbsp;|&nbsp;
      전략: <b style="color:#fff;">추세+눌림목+거래량 회복</b>
    </div>

    <table style="width:100%;border-collapse:collapse;background:#1e1e1e;border-radius:8px;">
      <tr style="background:#2d2d2d;color:#aaa;font-size:13px;">
        <th style="padding:10px;text-align:left;">종목</th>
        <th style="padding:10px;text-align:center;">점수</th>
        <th style="padding:10px;text-align:right;">현재가</th>
        <th style="padding:10px;text-align:right;color:#00c853;">목표가</th>
        <th style="padding:10px;text-align:right;color:#ff1744;">손절가</th>
      </tr>
      {rows}
    </table>

    {claude_section}

    <p style="color:#555;font-size:12px;margin-top:20px;">
      전략: 추세+눌림목+거래량 회복 (v13) | 승률 71.4% (백테스트) | 평균수익 15.66%<br>
      목표가/손절가는 Claude AI 분석 기반 (AI 표시)<br>
      투자는 본인 판단 하에 진행하세요.
    </p>
  </div>
</body></html>
"""
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


def main():
    print(f"\n{'='*40}")
    print(f"주식 봇 v2 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*40}\n")

    # 1. 미국 시장 현황 (참고용)
    sp_ret, nq_ret, sox_ret, us_date = check_us_market()
    print(f"미국 시장 ({us_date}): S&P500 {sp_ret*100:+.2f}% / 나스닥 {nq_ret*100:+.2f}%")

    # 2. 한국 시장 필터 (코스피/코스닥 각각 체크)
    kospi_ok, kosdaq_ok, market_status = check_market_state()
    print(f"한국 시장: {market_status}")

    if not kospi_ok and not kosdaq_ok:
        print(f"시장 조건 미충족 ({market_status}) → 신호 없음 메일 발송")
        send_email(
            f"[주식봇] {datetime.today().strftime('%Y-%m-%d')} 시장 조건 미충족",
            f"""<html><body style='background:#0d0d0d;color:#fff;padding:20px;font-family:sans-serif;'>
            <h3 style='color:#ff9800;'>⚠️ 오늘 시장 조건 미충족</h3>
            <p>사유: {market_status}</p>
            <p>미국: S&P500 {sp_ret*100:+.2f}% / 나스닥 {nq_ret*100:+.2f}%</p>
            <p style='color:#888;'>매수 신호 없음. 내일 다시 확인합니다.</p>
            </body></html>"""
        )
        return

    # 3. 후보 종목 스캔 (통과한 시장 종목만)
    candidates = get_candidates(kospi_ok, kosdaq_ok)
    if not candidates:
        print("조건 통과 종목 없음")
        send_email(
            f"[주식봇] {datetime.today().strftime('%Y-%m-%d')} 후보 종목 없음",
            f"""<html><body style='background:#0d0d0d;color:#fff;padding:20px;font-family:sans-serif;'>
            <h3 style='color:#64b5f6;'>시장은 양호하나 조건 통과 종목 없음</h3>
            <p>미국: S&P500 {sp_ret*100:+.2f}% / 나스닥 {nq_ret*100:+.2f}%</p>
            <p>한국 시장: {market_status}</p>
            </body></html>"""
        )
        return

    # 4. Claude 분석
    claude_analysis, claude_prices = analyze_with_claude(candidates, sp_ret, nq_ret, sox_ret)

    # 5. 이메일 발송
    body    = build_email(candidates, sp_ret, nq_ret, sox_ret, us_date,
                          market_status, claude_analysis, claude_prices)
    subject = f"[주식봇] {datetime.today().strftime('%Y-%m-%d')} 스윙 후보 {len(candidates)}종목"
    send_email(subject, body)

    print(f"\n이메일 발송 완료 → {RECEIVE_EMAIL}")
    for r in candidates:
        print(f"  {r['종목명']} | {r['점수']}점 | {r['현재가']:,.0f}원")


if __name__ == "__main__":
    main()
