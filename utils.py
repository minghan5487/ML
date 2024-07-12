import yfinance as yf
import pandas as pd

def download_yahoo_finance_csv(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    if not df.empty:
        df.to_csv(f'{symbol}.csv')
        print(f"Data for {symbol} downloaded successfully from {start_date} to {end_date}")
    else:
        print(f"No data for {symbol} in the given date range from {start_date} to {end_date}")
    return not df.empty 
