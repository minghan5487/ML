import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 


data = pd.read_csv('D:\\ML\\ML\\ML\\2330.TW.csv')


features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='實際值')
plt.plot(y_pred, label='預測值')
plt.xlabel('樣本')
plt.ylabel('價格')
plt.title('2330預測')
plt.legend()
plt.show()

