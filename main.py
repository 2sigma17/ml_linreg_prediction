import configparser
# Numpy, Pandas, Requests
import numpy as np
import pandas as pd
import requests
# Matplotlib
import matplotlib.pyplot as plt

# Using linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
config = configparser.ConfigParser()
config.read('alpha_vantage.cfg')
key = config['alpha_vantage']['key']

API_URL= "https://www.alphavantage.co/query"

# API payload
payload = {
    "function": "DIGITAL_CURRENCY_DAILY",
    "symbol": "BTC",
    "market": "USD",
    "apikey": key
}

res = requests.get(API_URL, payload)
data = res.json()
df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient="index")

# Convert df types into float
df = df.astype(float)

# Define exploratory variables (features we use to predict price of btc/usd)
df['ma13'] = df['4a. close (USD)'].shift(1).rolling(window=13).mean()
df['ma30'] = df['4a. close (USD)'].shift(1).rolling(window=30).mean()
df['values'] = df['4a. close (USD)']
# Drop NaN values
df = df.dropna()

# Initialising X and assigning the two feature variables
x = df[['ma13', 'ma30']]

# Getting the head of the data
# print(x.head())

# Setting up the dependent variable
y = df['values']

# Getting the head of data
# print(y.head())

# Set training data to 80% of data
training = 0.8
t = int(training*len(df))

# Training dataset
x_train = x[:t]
y_train = y[:t]

# Testing dataset
x_test = x[t:]
y_test = x[t:]

# Train model
model = LinearRegression().fit(x_train, y_train)
# Get list of predicted prices
pred_price = model.predict(x_test)
# Create new dataframe with list of predicted prices
pred_price = pd.DataFrame(pred_price, index=y_test.index, columns=['pred_price'])
# Add actual prices to dataframe
pred_price['actual_price'] = df['values']
# print(pred_price.head())
# Drop NaN values
pred_price.dropna()

# pred_price.plot(figsize=(10,5))
# y_test.plot()
# plt.legend(['Predicted Price', 'Actual Price'])
# plt.ylabel('Price of BTC in USD')
# plt.show()

# Computing accuracy of model
r_squared_score = r2_score(pred_price['actual_price'], pred_price['pred_price'])
print(f"The model has a r squared score of {r_squared_score}")
