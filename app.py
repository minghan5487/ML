import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, url_for
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from utils import download_history

app = Flask(__name__)

FEATURES = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10']
TARGET = 'Close'

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'PingFang TC', 'Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False


def build_features(df):
    df = df.copy()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df = df.dropna(subset=FEATURES + [TARGET])
    return df[FEATURES], df[TARGET]


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def save_plot(symbol, y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='實際值')
    plt.plot(y_test.index, y_pred, label='預測值')
    plt.xlabel('日期')
    plt.ylabel('收盤價')
    plt.title(f'{symbol} 收盤價預測')
    plt.legend()
    plt.tight_layout()
    os.makedirs(app.static_folder, exist_ok=True)
    plt.savefig(os.path.join(app.static_folder, 'prediction.png'))
    plt.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['stock_symbol'].strip().upper()
    try:
        years = int(request.form['years'])
    except ValueError:
        return render_template('index.html', error='請輸入有效的年數')

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    df = download_history(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if df.empty:
        return render_template('index.html', error=f'查無 {symbol} 的資料（台股請加 .TW，例如 2330.TW）')

    X, y = build_features(df)
    if len(X) < 30:
        return render_template('index.html', error='資料量不足，請拉長年數')

    y_test, y_pred, mse, r2 = train_and_evaluate(X, y)
    save_plot(symbol, y_test, y_pred)

    return render_template('result.html', symbol=symbol, mse=round(mse, 2), r2=round(r2, 4),
                           n_train=len(X) - len(y_test), n_test=len(y_test),
                           image_url=url_for('static', filename='prediction.png', v=int(time.time())))


if __name__ == '__main__':
    app.run(debug=True)
