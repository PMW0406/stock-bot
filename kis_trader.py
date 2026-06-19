"""
한국투자증권 자동매매 모듈
- 접근토큰 발급/캐싱 (24시간)
- 잔고 조회
- 시장가 매수/매도
- 보유종목 조회
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

APP_KEY    = os.environ["KIS_APP_KEY"]
APP_SECRET = os.environ["KIS_APP_SECRET"]
ACCOUNT    = os.environ["KIS_ACCOUNT"]  # 예: 44767363

BASE_URL   = "https://openapi.koreainvestment.com:9443"
TOKEN_FILE = "/tmp/kis_token.json"


def get_token():
    """접근토큰 발급 (캐싱, 24시간 유효)"""
    # 캐시된 토큰 확인
    if Path(TOKEN_FILE).exists():
        with open(TOKEN_FILE) as f:
            cached = json.load(f)
        expire = datetime.fromisoformat(cached["expires"])
        if datetime.now() < expire - timedelta(minutes=10):
            return cached["token"]

    # 새 토큰 발급
    url  = f"{BASE_URL}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey":     APP_KEY,
        "appsecret":  APP_SECRET,
    }
    res = requests.post(url, json=body)
    res.raise_for_status()
    data  = res.json()
    token = data["access_token"]

    # 캐싱
    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "token":   token,
            "expires": (datetime.now() + timedelta(hours=23)).isoformat()
        }, f)

    return token


def get_headers(tr_id):
    return {
        "content-type":  "application/json",
        "authorization": f"Bearer {get_token()}",
        "appkey":        APP_KEY,
        "appsecret":     APP_SECRET,
        "tr_id":         tr_id,
    }


def get_balance():
    """주식 잔고 조회"""
    url    = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
    params = {
        "CANO":           ACCOUNT,
        "ACNT_PRDT_CD":  "01",
        "AFHR_FLPR_YN":  "N",
        "OFL_YN":        "",
        "INQR_DVSN":     "02",
        "UNPR_DVSN":     "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN":     "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    res  = requests.get(url, headers=get_headers("TTTC8434R"), params=params)
    data = res.json()

    holdings = []
    for item in data.get("output1", []):
        qty = int(item.get("hldg_qty", 0))
        if qty > 0:
            holdings.append({
                "종목코드": item["pdno"],
                "종목명":   item["prdt_name"],
                "보유수량": qty,
                "평균단가": float(item["pchs_avg_pric"]),
                "현재가":   float(item["prpr"]),
                "평가손익": float(item["evlu_pfls_amt"]),
                "수익률":   float(item["evlu_pfls_rt"]),
            })

    cash = float(data.get("output2", [{}])[0].get("dnca_tot_amt", 0))
    return holdings, cash


def buy_market(ticker, qty):
    """시장가 매수"""
    url  = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
    body = {
        "CANO":         ACCOUNT,
        "ACNT_PRDT_CD": "01",
        "PDNO":         ticker,
        "ORD_DVSN":     "01",   # 시장가
        "ORD_QTY":      str(qty),
        "ORD_UNPR":     "0",
    }
    res  = requests.post(url, headers=get_headers("TTTC0802U"), json=body)
    data = res.json()
    ok   = data.get("rt_cd") == "0"
    return ok, data.get("msg1", "")


def sell_market(ticker, qty):
    """시장가 매도"""
    url  = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
    body = {
        "CANO":         ACCOUNT,
        "ACNT_PRDT_CD": "01",
        "PDNO":         ticker,
        "ORD_DVSN":     "01",   # 시장가
        "ORD_QTY":      str(qty),
        "ORD_UNPR":     "0",
        "SLL_BUY_DVSN_CD": "01",  # 매도
    }
    res  = requests.post(url, headers=get_headers("TTTC0801U"), json=body)
    data = res.json()
    ok   = data.get("rt_cd") == "0"
    return ok, data.get("msg1", "")


def get_current_price(ticker):
    """현재가 조회"""
    url    = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
    res    = requests.get(url, headers=get_headers("FHKST01010100"), params=params)
    data   = res.json()
    return float(data["output"]["stck_prpr"])


if __name__ == "__main__":
    print("잔고 조회 테스트...")
    holdings, cash = get_balance()
    print(f"예수금: {cash:,.0f}원")
    print(f"보유종목: {len(holdings)}개")
    for h in holdings:
        print(f"  {h['종목명']} {h['보유수량']}주 | 수익률 {h['수익률']:.1f}%")
