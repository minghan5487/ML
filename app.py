import os
import time

import matplotlib
matplotlib.use('Agg')  # 伺服器端繪圖，不開視窗
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, url_for
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from utils import download_history

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

FEATURES = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10']
TARGET = 'Close'

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'PingFang TC', 'Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False


def build_features(data):
    data = data.copy()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data = data.dropna(subset=FEATURES + [TARGET])
    return data[FEATURES], data[TARGET]


def train_and_evaluate(X, y):
    # 時間序列不打亂：以前 80% 訓練、後 20% 測試，避免用未來資料預測過去
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def save_plot(symbol, y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='實際值', color='#1f77b4')
    plt.plot(y_test.index, y_pred, label='預測值', color='#ff7f0e')
    plt.xlabel('日期')
    plt.ylabel('收盤價')
    plt.title(f'{symbol} 收盤價預測（測試區間）')
    plt.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs(STATIC_DIR, exist_ok=True)
    plt.savefig(os.path.join(STATIC_DIR, 'prediction.png'))
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
    data = download_history(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if data.empty:
        return render_template('index.html', error=f'查無 {symbol} 在此期間的資料（台股請加 .TW，例如 2330.TW）')

    X, y = build_features(data)
    if len(X) < 30:
        return render_template('index.html', error='資料量不足，請拉長年數')

    y_test, y_pred, mse, r2 = train_and_evaluate(X, y)
    save_plot(symbol, y_test, y_pred)

    image_url = url_for('static', filename='prediction.png', v=int(time.time()))
    return render_template('result.html', symbol=symbol, mse=round(mse, 2), r2=round(r2, 4),
                           n_train=len(X) - len(y_test), n_test=len(y_test), image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
