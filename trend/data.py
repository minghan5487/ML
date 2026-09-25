import time
from datetime import date

import pandas as pd
import requests
import yfinance as yf

from trend.config import DATA_DIR, MARKET, START_DATE, STOCKS, VALUATION_DIR

FINMIND_URL = 'https://api.finmindtrade.com/api/v4/data'


def path_for(symbol):
    return DATA_DIR / f"{symbol.replace('^', '')}.csv"


def is_fresh(path):
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()


def download_valuation(code):
    resp = requests.get(FINMIND_URL, params={'dataset': 'TaiwanStockPER', 'data_id': code, 'start_date': START_DATE},
                        timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()['data'])
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')[['PER', 'PBR', 'dividend_yield']]


def download(symbol, retries=3):
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(start=START_DATE, auto_adjust=True)
            if not df.empty:
                df.index = df.index.tz_localize(None).normalize()
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f'  {symbol} 第 {attempt + 1} 次失敗：{e}')
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def update_all(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VALUATION_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [MARKET] + [f'{code}.TW' for code in STOCKS]

    for i, symbol in enumerate(symbols, 1):
        path = path_for(symbol)
        if not force and is_fresh(path):
            continue
        df = download(symbol)
        if df.empty:
            print(f'[{i}/{len(symbols)}] {symbol} 下載失敗，沿用舊資料')
            continue
        df.to_csv(path)
        print(f'[{i}/{len(symbols)}] {symbol} {len(df)} 筆，最新 {df.index[-1].date()}')

    for i, code in enumerate(STOCKS, 1):
        path = VALUATION_DIR / f'{code}.csv'
        if not force and is_fresh(path):
            continue
        try:
            df = download_valuation(code)
            df.to_csv(path)
            print(f'[估值 {i}/{len(STOCKS)}] {code} {len(df)} 筆')
        except Exception as e:
            print(f'[估值 {i}/{len(STOCKS)}] {code} 下載失敗：{e}')
        time.sleep(0.5)


def load(symbol):
    path = path_for(symbol)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_valuation(code):
    path = VALUATION_DIR / f'{code}.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == '__main__':
    update_all()
