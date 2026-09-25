# 台股走勢預測

用 scikit-learn 決策樹迴歸預測股票收盤價的 Flask 網頁。輸入股票代碼與回溯年數，會自動下載歷史資料、建立特徵、訓練模型並畫出預測結果。

![預測結果](docs/prediction-example.png)

## 做法

1. 用 yfinance 下載日線資料
2. 特徵：開盤、最高、最低、成交量、5 日與 10 日均線；目標：收盤價
3. 依時間順序切分，前 80% 訓練、後 20% 測試
4. DecisionTreeRegressor 訓練，以 MSE、R² 評估
5. 畫出測試區間的實際值與預測值

## 技術

Python、scikit-learn、pandas、matplotlib、yfinance、Flask

## 執行

```bash
pip install -r requirements.txt
python app.py
```

開啟 http://127.0.0.1:5000 ，輸入代碼（台股加 .TW，例如 2330.TW）與年數。

## 待改進

- 目前是用當天的開高低價預測當天收盤，之後改成用前一天的資料預測隔天
- 用 TimeSeriesSplit 做交叉驗證，並比較隨機森林、XGBoost
- 和 [money](https://github.com/minghan5487/money) 整合
