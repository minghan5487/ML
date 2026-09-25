# 台股走勢預測

[![每日訓練](https://github.com/minghan5487/ML/actions/workflows/train.yml/badge.svg)](https://github.com/minghan5487/ML/actions/workflows/train.yml)

預測 0050 成分股未來 20 個交易日的漲跌機率。GitHub Actions 每個交易日收盤後自動下載資料、重新訓練、驗證，最新結果見 **[reports/latest.md](reports/latest.md)**。

## 流程

1. **資料**：yfinance 下載 50 檔個股與加權指數日線（2012 起），FinMind 下載每日本益比、淨值比、殖利率
2. **特徵**：技術指標，加上依投資專家公開方法論量化的因子
3. **標籤**：未來 20 個交易日報酬 > 0
4. **驗證**：Walk-forward，5 個各 250 日的測試區間，訓練與測試之間間隔 20 日避免標籤重疊造成資訊洩漏
5. **模型選擇**：Logistic Regression、Random Forest、HistGradientBoosting 比較 AUC，新模型不比舊模型差才替換
6. **輸出**：個股上漲機率、各專家觀點、特徵重要性、每次訓練紀錄

## 專家因子

| 專家 | 方法 | 特徵 |
|---|---|---|
| Meb Faber | 趨勢跟隨、動能 | 200 日均線乖離、12-1 月報酬 |
| Aswath Damodaran | 估值 | 本益比、淨值比、殖利率，與自身 3 年中位數比較 |
| Charlie Bilello | 市場廣度 | 權值股站上 200 日線比例 |
| Morgan Housel | 長期心態 | 大盤距一年高點回檔幅度、大盤 200 日乖離 |
| Peter Brandt | 技術突破 | 距 52 週高點、20 日 VWAP 乖離 |

## 目前結果

以 2021–2026 五個年度測試：

- AUC 約 0.55（隨機為 0.50）；只用技術指標約 0.53，加入專家因子後提升
- 最重要的特徵是大盤距一年高點的回檔幅度、200 日均線乖離、距 52 週高點
- 模型挑出的前 20% 股票，5 個年度中有 4 年的 20 日平均報酬高於全體平均
- 準確率約 57%，但與「全猜上漲」接近，因為這段期間台股多數時間上漲，所以以 AUC 與選股報酬評估較有意義

## 執行

```bash
pip install -r requirements.txt
python -m trend.train
python app.py
```

開啟 http://127.0.0.1:5000 ，輸入股票代碼即可看到明日預測收盤價；`/trend` 是 0050 成分股 20 日走勢儀表板。

## 專案結構

```
trend/config.py     股票清單與參數
trend/data.py       下載股價與估值資料
trend/features.py   特徵工程與專家觀點規則
trend/train.py      Walk-forward 驗證、模型選擇、產生報告
reports/            每次訓練的結果（Actions 自動更新）
.github/workflows/  每日訓練排程
app.py              儀表板與隔日價格預測
```

## 限制

- 成分股清單為手動整理，未隨 0050 調整自動更新，且只包含目前的成分股，有存活者偏差
- 未計入交易成本
- 僅為機器學習練習，不構成投資建議；專家觀點為依其公開方法論量化的規則，並非本人意見
