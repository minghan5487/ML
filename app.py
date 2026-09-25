import json
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

from trend.config import MODEL_DIR, REPORT_DIR
from utils import download_history

app = Flask(__name__)

EXPERTS = {
    'Meb Faber': '趨勢跟隨',
    'Aswath Damodaran': '估值',
    'Charlie Bilello': '市場廣度',
    'Morgan Housel': '長期心態',
    'Peter Brandt': '技術突破',
}
FEATURE_LABELS = {
    'mkt_drawdown_250': '大盤距一年高點（Housel）', 'faber_trend_200': '200 日均線乖離（Faber）',
    'high_52w_gap': '距 52 週高點（Brandt）', 'vwap_gap_20': '20 日 VWAP 乖離（Brandt）',
    'mom_12_1': '12-1 月動能（Faber）', 'breadth_200': '站上 200 日線比例（Bilello）',
    'mkt_trend_200': '大盤 200 日乖離（Housel）', 'per': '本益比（Damodaran）', 'pbr': '淨值比（Damodaran）',
    'dividend_yield': '殖利率（Damodaran）', 'per_vs_3y': '本益比 vs 3 年中位數（Damodaran）',
    'pbr_vs_3y': '淨值比 vs 3 年中位數（Damodaran）', 'ret_5': '5 日報酬', 'ret_20': '20 日報酬',
    'ret_60': '60 日報酬', 'ma_gap_20': '20 日均線乖離', 'ma_gap_60': '60 日均線乖離', 'rsi_14': 'RSI(14)',
    'macd_hist': 'MACD 柱狀體', 'volatility_20': '20 日波動率', 'volume_ratio': '量比',
    'hl_range': '平均振幅', 'rel_strength_20': '相對大盤強弱',
}

FEATURES = ['return', 'ma5_gap', 'ma10_gap', 'hl_range', 'oc_change', 'volume_change']

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'PingFang TC', 'Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False


def read_csv(name, **kwargs):
    path = REPORT_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


@app.route('/')
def home():
    meta_path = MODEL_DIR / 'meta.json'
    if not meta_path.exists():
        return render_template('trend.html', meta=None)

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    predictions = read_csv('predictions.csv', dtype={'code': str})
    history = read_csv('history.csv')
    folds = read_csv('folds.csv')
    importance = read_csv('importance.csv').head(10)
    importance['label'] = importance['feature'].map(FEATURE_LABELS).fillna(importance['feature'])

    stocks = []
    for _, r in predictions.iterrows():
        stocks.append({
            'code': r.code, 'name': r['name'], 'close': r.close, 'ret_20': r.ret_20, 'prob_up': r.prob_up,
            'views': [(e, r[f'{e}|signal'], r[f'{e}|reason']) for e in EXPERTS],
        })

    consensus = {e: predictions[f'{e}|signal'].value_counts().to_dict() for e in EXPERTS}
    return render_template('trend.html', meta=meta, stocks=stocks, experts=EXPERTS, consensus=consensus,
                           history=history.to_dict('records'), folds=folds.to_dict('records'),
                           importance=importance.to_dict('records'), data_date=predictions['date'].iloc[0])


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


@app.route('/regression')
def regression():
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
