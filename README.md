# 台股走勢預測

用 scikit-learn 決策樹迴歸預測隔日收盤價的 Flask 網頁。輸入股票代碼與回溯年數，會自動下載歷史資料、建立特徵、訓練模型並畫出預測結果。

![預測結果](docs/prediction-example.png)

## 做法

1. 用 yfinance 下載日線資料
2. 特徵：當日報酬率、與 5 日 / 10 日均線的乖離、當日振幅、開收盤變化、成交量變化
3. 目標：隔日報酬率，再換算回隔日收盤價
4. 依時間順序切分，前 80% 訓練、後 20% 測試
5. DecisionTreeRegressor（max_depth=5）訓練，以 MSE、R²、漲跌方向準確率評估

一開始直接預測收盤價，但決策樹無法預測超出訓練資料範圍的價格，股價創新高時會一直低估（2330 測試 R² 為負）。改成預測報酬率後，特徵與目標都不受股價水準影響，2330 三年資料測試 R² 約 0.95、方向準確率約 58%。

## 技術

Python、scikit-learn、pandas、matplotlib、yfinance、Flask

## 執行

```bash
pip install -r requirements.txt
python app.py
```

開啟 http://127.0.0.1:5000 ，輸入代碼（台股加 .TW，例如 2330.TW）與年數。

## 待改進

- R² 高主要來自隔日價格接近當日價格，方向準確率才是較有參考價值的指標，之後加入更多特徵提升
- 用 TimeSeriesSplit 做交叉驗證，並比較隨機森林、XGBoost
- 和 [money](https://github.com/minghan5487/money) 整合
