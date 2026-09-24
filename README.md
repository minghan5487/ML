# 🌳 台股走勢預測（Decision Tree）

以 **scikit-learn 決策樹迴歸** 預測股票收盤價的 Flask 網頁應用：輸入股票代碼與回溯年數，自動下載歷史資料、建立特徵、訓練模型並繪出預測結果。

> 個人專案｜機器學習、資料前處理、視覺化

![預測結果範例](docs/prediction-example.png)

## 流程

```
輸入代碼與年數 → yfinance 下載歷史資料 → 特徵工程 → 依時間順序切分 → DecisionTreeRegressor → MSE / R² 評估 → 繪圖輸出
```

- **特徵**：開盤價、最高價、最低價、成交量、5 日均線（MA5）、10 日均線（MA10）
- **目標**：收盤價
- **切分方式**：依時間順序，前 80% 訓練、後 20% 測試（不打亂，避免以未來資料訓練）
- **評估**：Mean Squared Error、R² Score，結果頁同時顯示訓練 / 測試筆數
- **視覺化**：matplotlib 繪製測試區間的實際值與預測值比較圖

## 技術

`Python` · `scikit-learn` · `pandas` · `NumPy` · `matplotlib` · `yfinance` · `Flask`

## 專案結構

```
app.py             # Flask 主程式：特徵工程、訓練評估、繪圖
utils.py           # yfinance 歷史資料下載
templates/         # 輸入頁與結果頁
static/            # 執行時產生的預測圖
data/2330.TW.csv   # 範例資料（台積電）
docs/              # README 範例圖
```

## 執行方式

```bash
pip install -r requirements.txt
python app.py
```

開啟 `http://127.0.0.1:5000`，輸入代碼（台股請加 `.TW`，例如 `2330.TW`）與年數。

## 後續規劃

- 與 [money｜台股查詢平台](https://github.com/minghan5487/money) 整合，同時提供即時資訊與趨勢預測
- 目前以當日開高低價預測當日收盤，下一步改為以前一日資料預測隔日收盤，更貼近實際交易情境
- 導入 TimeSeriesSplit 交叉驗證，並比較隨機森林、XGBoost 等模型

---
作者：[吳明翰](https://minghan5487.github.io/my/)
