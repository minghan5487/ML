import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默認字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 讀取數據
data = pd.read_csv('D:\\ML\\ML\\ML\\2330.TW.csv')

# 選擇特徵和目標變量
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features]
y = data[target]

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練決策樹回歸模型
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# 視覺化結果
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='實際值')
plt.plot(y_pred, label='預測值')
plt.xlabel('樣本')
plt.ylabel('價格')
plt.title('2330預測')
plt.legend()
plt.show()

