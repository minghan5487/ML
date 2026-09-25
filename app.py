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

FEATURES = ['return', 'ma5_gap', 'ma10_gap', 'hl_range', 'oc_change', 'volume_change']

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'PingFang TC', 'Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False


def build_features(df):
    close = df['Close']
    df = df.assign(
        **{
            'return': close.pct_change(),
            'ma5_gap': close / close.rolling(5).mean() - 1,
            'ma10_gap': close / close.rolling(10).mean() - 1,
            'hl_range': (df['High'] - df['Low']) / close,
            'oc_change': (close - df['Open']) / df['Open'],
            'volume_change': df['Volume'].pct_change(),
            'next_close': close.shift(-1),
        }
    )
    df['target'] = df['next_close'] / close - 1
    df = df.replace([float('inf'), float('-inf')], float('nan')).dropna(subset=FEATURES + ['target'])
    return df


def train_and_evaluate(df):
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)
    model.fit(train[FEATURES], train['target'])

    pred_return = model.predict(test[FEATURES])
    actual = test['next_close']
    predicted = pd.Series(test['Close'].values * (1 + pred_return), index=test.index)

    direction_acc = ((pred_return > 0) == (test['target'] > 0)).mean()
    return actual, predicted, mean_squared_error(actual, predicted), r2_score(actual, predicted), direction_acc, len(train)


def save_plot(symbol, actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='實際值')
    plt.plot(predicted.index, predicted.values, label='預測值')
    plt.xlabel('日期')
    plt.ylabel('收盤價')
    plt.title(f'{symbol} 隔日收盤價預測')
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

    data = build_features(df)
    if len(data) < 100:
        return render_template('index.html', error='資料量不足，請拉長年數')

    actual, predicted, mse, r2, direction_acc, n_train = train_and_evaluate(data)
    save_plot(symbol, actual, predicted)

    return render_template('result.html', symbol=symbol, mse=round(mse, 2), r2=round(r2, 4),
                           direction_acc=f'{direction_acc:.1%}', n_train=n_train, n_test=len(actual),
                           image_url=url_for('static', filename='prediction.png', v=int(time.time())))


if __name__ == '__main__':
    app.run(debug=True)
