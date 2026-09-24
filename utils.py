import os

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def download_history(symbol, start_date, end_date):
    """以 yfinance 下載日線歷史資料，並存一份 CSV 至 data/。查無資料時回傳空的 DataFrame。"""
    df = yf.Ticker(symbol).history(start=start_date, end=end_date)
    if df.empty:
        return pd.DataFrame()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, f'{symbol}.csv'))
    return df
