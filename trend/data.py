import time
from datetime import date

import pandas as pd
import requests
import yfinance as yf

from trend.config import (CHIP_DIR, DATA_DIR, MARGIN_DIR, MARKET, REVENUE_DIR, REVENUE_PUBLISH_LAG_DAYS,
                          START_DATE, STOCKS, VALUATION_DIR)

FINMIND_URL = 'https://api.finmindtrade.com/api/v4/data'


def path_for(symbol):
    return DATA_DIR / f"{symbol.replace('^', '')}.csv"


def is_fresh(path):
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()


def _finmind(dataset, code):
    """向 FinMind 取一個 dataset；失敗或無資料一律回空 DataFrame，由呼叫端決定不落盤。"""
    resp = requests.get(FINMIND_URL,
                        params={'dataset': dataset, 'data_id': code, 'start_date': START_DATE},
                        timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return pd.DataFrame(payload.get('data') or [])


def download_chip(code):
    """三大法人買賣超（長格式：每天每種法人一列），轉成每日淨買超股數。"""
    df = _finmind('TaiwanStockInstitutionalInvestorsBuySell', code)
    if df.empty or not {'date', 'name', 'buy', 'sell'} <= set(df.columns):
        return pd.DataFrame()
    df['net'] = df['buy'].astype(float) - df['sell'].astype(float)
    wide = df.pivot_table(index='date', columns='name', values='net', aggfunc='sum')
    wide.index = pd.to_datetime(wide.index)
    keep = {'Foreign_Investor': 'foreign_net', 'Investment_Trust': 'trust_net'}
    out = pd.DataFrame(index=wide.index)
    for src, dst in keep.items():
        out[dst] = wide[src] if src in wide.columns else float('nan')
    return out.sort_index()


def download_margin(code):
    """融資融券餘額。"""
    df = _finmind('TaiwanStockMarginPurchaseShortSale', code)
    need = {'date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance'}
    if df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    out = df.set_index('date')[['MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']].astype(float)
    out.columns = ['margin_balance', 'short_balance']
    return out.sort_index()


def download_revenue(code):
    """月營收。index 設為『最早可用日』＝資料期末 + REVENUE_PUBLISH_LAG_DAYS，避免前視偏誤。"""
    df = _finmind('TaiwanStockMonthRevenue', code)
    if df.empty or not {'date', 'revenue'} <= set(df.columns):
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    out = df.set_index('date')[['revenue']].astype(float).sort_index()
    out['available_at'] = out.index + pd.Timedelta(days=REVENUE_PUBLISH_LAG_DAYS)
    return out


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

    _fetch_finmind_set('估值', VALUATION_DIR, download_valuation, force)
    _fetch_finmind_set('籌碼', CHIP_DIR, download_chip, force)
    _fetch_finmind_set('融資券', MARGIN_DIR, download_margin, force)
    _fetch_finmind_set('月營收', REVENUE_DIR, download_revenue, force)


def _fetch_finmind_set(label, outdir, fn, force):
    outdir.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0
    for i, code in enumerate(STOCKS, 1):
        path = outdir / f'{code}.csv'
        if not force and is_fresh(path):
            skipped += 1
            continue
        try:
            df = fn(code)
            if df is None or df.empty:
                failed += 1
            else:
                df.to_csv(path)
                ok += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f'  [{label}] {code} 下載失敗：{str(e)[:80]}')
        time.sleep(0.5)
    print(f'[{label}] 成功 {ok}、沿用既有 {skipped}、無資料或失敗 {failed}')


def load(symbol):
    path = path_for(symbol)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_csv(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f'  {path.name} 讀取失敗（{type(e).__name__}），視為無資料')
        return None


def load_chip(code):
    return _load_csv(CHIP_DIR / f'{code}.csv')


def load_margin(code):
    return _load_csv(MARGIN_DIR / f'{code}.csv')


def load_revenue(code):
    return _load_csv(REVENUE_DIR / f'{code}.csv')


def load_valuation(code):
    path = VALUATION_DIR / f'{code}.csv'
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f'  估值檔 {code} 讀取失敗（{type(e).__name__}），視為無估值資料')
        return None


if __name__ == '__main__':
    update_all()
