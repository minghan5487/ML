from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from utils import download_yahoo_finance_csv
import os

app = Flask(__name__)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol']
    years = int(request.form['years'])
    
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    data_downloaded = download_yahoo_finance_csv(stock_symbol, start_date, end_date)
    if not data_downloaded:
        print(f"No data available for {stock_symbol} in the date range from {start_date} to {end_date}")
        return "No data available for the specified date range."

    data = pd.read_csv(f'{stock_symbol}.csv')
    if data.empty:
        print(f"No data available in the CSV file for {stock_symbol}")
        return "No data available in the CSV file."

    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()

    features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10']
    target = 'Close'

    X = data[features].dropna()
    y = data[target][len(data) - len(X):]

    if len(X) == 0 or len(y) == 0:
        print(f"No data available after processing for {stock_symbol}")
        return "No data available after processing."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(y_train) == 0:
        print(f"Training set is empty for {stock_symbol}. Adjust the date range or check data availability.")
        return "Training set is empty. Adjust the date range or check data availability."

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(12, 8))  
    plt.plot(y_test.values, label='實際值', color='blue')
    plt.plot(y_pred, label='預測值', color='orange')
    plt.xlabel('樣本')  
    plt.ylabel('價格')  
    plt.title('2330分析') 
    plt.legend(loc='upper right')  
    plt.savefig('static/prediction.png')
    plt.close()  
    
    print(f"Prediction plot saved for {stock_symbol}")
    
    return render_template('result.html', mse=mse, r2=r2, image_url='static/prediction.png')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
