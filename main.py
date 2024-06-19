# %%
import warnings
warnings.filterwarnings('ignore')

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Technical Analysis
from talib import RSI, BBANDS, ATR, NATR, MACD

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier


import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import joblib

# neural network
from tqdm import tqdm
import torch
import torch.nn as nn
  # our model
from LSTM1 import StockLSTM1
from LSTM1 import LSTM1_stock_predict
from LSTM1 import LSTMdataset1
from LSTM2 import StockLSTM2
from LSTM2 import LSTM2_stock_predict
from LSTM2 import LSTMdataset2
from GRU1 import StockGRU1
from GRU1 import GRU1_stock_predict
from GRU1 import GRUdataset1
from GRU2 import StockGRU2
from GRU2 import GRU2_stock_predict
from GRU2 import GRUdataset2
# Feature important
import FeatureImportance

# %% [markdown]
# # Data Preprocessing

# %%
# def read_tickers_sp500(file_path):
#     with open(file_path, 'r') as file:
#         tickers_sp500 = [line.strip() for line in file]
#     return tickers_sp500

# tickers_sp500 = read_tickers_sp500('tickers_sp500.txt')
# print(tickers_sp500)

# %%
# Download data

# Stocks list
# top 7 MarketCap in S&P500(^GSPC)
# tickers = ['^GSPC', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'META']
tickers = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOG', 'BRK.B', 'LLY', 'JPM', 'AVGO', 'XOM', 'UNH', 'V', 'TSLA', 'PG', 'MA', 'JNJ', 'HD', 'MRK', 'COST', 'ABBV', 'CVX', 'CRM', 'BAC', 'NFLX']
# tickers = tickers_sp500
start_date = '2014-05-01'
end_date = '2024-05-01'

df_prices_download = yf.download(tickers=tickers, start=start_date, end=end_date, group_by='ticker')

# %% [markdown]
# 

# %%
# Format into large table
# col: OHLCV
# rows(multi-index): Ticker, Date

df_prices = df_prices_download.stack(level=0, dropna=False)
df_prices = df_prices.swaplevel(0, 1)
df_prices = df_prices.loc[tickers].sort_index(level='Ticker')
df_prices.dropna(inplace=True)

# Use 'Adj Close' instead of 'Close'
df_prices.drop('Close', axis=1, inplace=True)
df_prices.rename(columns={'Adj Close': 'Close'}, inplace=True)

df_prices

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ### RSI - Relative Strength Index
# RSI compares the magnitude of recent price changes across stocks to identify stocks as overbought or oversold.

# %%
rsi = df_prices.groupby(level='Ticker', group_keys=False).Close.apply(RSI)
df_prices['RSI'] = rsi

# %% [markdown]
# ### Bollinger Bands
# Bollinger Bands is a technical analysis tool used to determine where prices are high and low relative to each other.

# %%
def compute_bb(close):
    high, mid, low = BBANDS(np.log1p(close), timeperiod=20)
    return pd.DataFrame({'BB_High': high,
                         'BB_Mid': mid, # SMA20
                         'BB_Low': low},
                        index=close.index)

bbands = df_prices.groupby(level='Ticker', group_keys=False).Close.apply(compute_bb)
df_prices = pd.concat([df_prices, bbands], axis=1)

# %% [markdown]
# ### ATR - Average True Range
# The average true range (ATR) indicator shows the volatility of the market.

# %%
by_ticker = df_prices.groupby('Ticker', group_keys=False)

def compute_atr(stock_data):
    atr = ATR(stock_data.High,
              stock_data.Low,
              stock_data.Close,
              timeperiod=14)
    return atr.sub(atr.mean()).div(atr.std())

df_prices['ATR'] = by_ticker.apply(compute_atr)
# Normalized Average True Range (NATR)
df_prices['NATR'] = by_ticker.apply(lambda x: NATR(high=x.High, low=x.Low, close=x.Close))

# %% [markdown]
# ### MACD - Moving Average Convergence/Divergence

# %%
def compute_macd(close):
    macd = MACD(close)[0]
    return macd.sub(macd.mean()).div(macd.std())

df_prices['MACD'] = df_prices.groupby(level='Ticker', group_keys=False).Close.apply(compute_macd)

# %% [markdown]
# ## Determine Investment Universe

# %% [markdown]
# ### Dollar Volume

# %%
# Close: USD
# Volumn: Amount
df_prices['Dollar_Volume'] = (df_prices.loc[:, 'Close']
                           .mul(df_prices.loc[:, 'Volume'], axis=0))

df_prices.Dollar_Volume /= 1e6 # Dollar_Volume: Million USD

df_prices.dropna(inplace=True)

# %%
remian_cols = [c for c in df_prices.columns.unique(0) if c not in ['Dollar_Volume', 'Volume']]

# New data frame: 'data' - load to model
data = (
    pd.concat(
        [
        # avg(1M) Dollar_Volume
            df_prices.unstack("Ticker")
            .Dollar_Volume.resample('D')
            .mean()
            .stack("Ticker")
            .to_frame("Dollar_Volume"),
        # (Adj)Close & Technical Indicators
            df_prices.unstack("Ticker")[remian_cols]
            .resample('D')
            .last()
            .stack("Ticker")
        ],
        axis=1
    )
    .swaplevel()
    .sort_index(level='Ticker')
    .dropna()
)

data.info()

# %% [markdown]
# ## Monthly Return

# %%
outlier_cutoff = 0.01 # winsorize returns at the [1%, 99%]
# lags = [1, 3, 6, 12] # Month timeframe
lags = [1, 5, 10, 21, 42, 63] # Day timeframe
returns = []

for lag in lags:
    returns.append(data
                   .Close
                   .unstack('Ticker')
                   .sort_index()
                   .pct_change(lag)
                   .stack('Ticker')
                   .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                          upper=x.quantile(1-outlier_cutoff)))
                   .add(1)
                   .pow(1/lag)
                   .sub(1)
                   .to_frame(f'Return_{lag}d')
                   )

returns.append(data.High.unstack('Ticker').sort_index().pct_change(1).stack('Ticker')
               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),upper=x.quantile(1-outlier_cutoff)))
               .to_frame("High_Return"))
returns.append(data.Low.unstack('Ticker').sort_index().pct_change(1).stack('Ticker')
               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),upper=x.quantile(1-outlier_cutoff)))
               .to_frame("Low_Return"))
returns.append(data.Open.unstack('Ticker').sort_index().pct_change(1).stack('Ticker')
               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),upper=x.quantile(1-outlier_cutoff)))
               .to_frame("Open_Return"))

df_returns = pd.concat(returns, axis=1).swaplevel().sort_index(level='Ticker')
df_returns.info()

# %%
# merge returns -> data
# drop 'Close', use 'Returns' instead
data = data.join(df_returns).drop(['Close','High','Low','Open'], axis=1).dropna()
data.info()

# %% [markdown]
# ## Price Momentum
# This factor computes the total return for a given number of prior trading days d.

# %%
# # Month timeframe
# for lag in [3, 6, 12]:
#     data[f'Momentum_{lag}'] = data[f'Return_{lag}m'].sub(data.Return_1m) # 3Xm - 1m
#     if lag > 3:
#         data[f'Momentum_3_{lag}'] = data[f'Return_{lag}m'].sub(data.Return_3m) # 6Xm - 3m

# Day timeframe
for lag in [5, 10, 21, 42, 63]:
    data[f'Momentum_{lag}'] = data[f'Return_{lag}d'].sub(data.Return_1d) # 3Xm - 1m
    if lag > 5:
        data[f'Momentum_5_{lag}'] = data[f'Return_{lag}d'].sub(data.Return_5d) # 6Xm - 3m

data.info()

# %% [markdown]
# ## Date Indicators

# %%
dates = data.index.get_level_values('Date')
data['Year'] = dates.year
data['Month'] = dates.month

# %% [markdown]
# ## Target: Holding Period Returns
# 1 day target holding period\
# = to predict return in next 1 day (tomorrow)

# %%
data['target'] = data.groupby(level='Ticker')['Return_1d'].shift(-1)
data = data.dropna()
data.info()

# %% [markdown]
# ## Save data to local

# %%
# DATA_PATH = 'data'

# df_prices.to_csv(f'{DATA_PATH}/prices.csv', index=True)
# df_returns.to_csv(f'{DATA_PATH}/returns.csv', index=True)
# data.to_csv(f'{DATA_PATH}/data.csv', index=True)

# %%
df_prices.info()

# %%
df_returns.info()

# %%
data.info()

# %% [markdown]
# # Split Data for Trading Evaluation
# - **Train and Test Set:** 2014 - 2023 (9y)
# - **Trading Evaluation:** 2023 - 2024 (1y)

# %%
data_reset = data.reset_index()
data_reset['Date'] = pd.to_datetime(data_reset['Date'])

split_date = pd.to_datetime('2023-05-01')

df_traintest= data_reset[data_reset['Date'] < split_date]
df_evaluate = data_reset[data_reset['Date'] >= split_date]

df_traintest.set_index(['Ticker', 'Date'], inplace=True)
df_evaluate.set_index(['Ticker', 'Date'], inplace=True)

print(df_traintest.shape)
print(df_evaluate.shape)

# %%
df_traintest.tail()

# %%
df_evaluate.head()

# %% [markdown]
# # Scaling Data
# **Strategy:** Standard Score (Z-Score)

# %%
selected_data = df_traintest

norm_cols = selected_data.columns[:-1]  # Excluding Target column

# Create a ColumnTransformer to apply scaling only to specified columns
column_transformer = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), norm_cols)
    ],
    remainder='passthrough'  # Keep non-scaled columns unchanged
)

norm_data = column_transformer.fit_transform(selected_data)
norm_evaluate = column_transformer.fit_transform(df_evaluate)

print(norm_data)
print(norm_data.shape)
print(norm_evaluate.shape)

# %% [markdown]
# # Model

# %% [markdown]
# ## Train-Test Split
# **Strategy:** Simple Split
# 
# Split arrays or matrices into random train and test subsets.

# %% [markdown]
# ### Split for Regressors

# %%
norm_X = norm_data[:, :-1]
norm_y = norm_data[:, -1]

X = df_traintest.drop('target', axis=1)
y = df_traintest.target

norm_X_train, norm_X_test, norm_y_train, norm_y_test = train_test_split(norm_X, norm_y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print("Train Set:", X_train.shape, y_train.shape)
print("Test Set:", X_test.shape, y_test.shape)

# %% [markdown]
# ### Split for Classifiers

# %%
norm_X_bool = norm_data[:, :-1]
norm_y_bool = norm_data[:, -1] > 0

X_bool = df_traintest.drop('target', axis=1)
y_bool = df_traintest.target > 0

norm_X_train_bool, norm_X_test_bool, norm_y_train_bool, norm_y_test_bool = train_test_split(norm_X_bool, norm_y_bool, random_state=42)
X_train_bool, X_test_bool, y_train_bool, y_test_bool = train_test_split(X_bool, y_bool, random_state=42)

print("Train Set:", X_train_bool.shape, y_train_bool.shape)
print("Test Set:", X_test_bool.shape, y_test_bool.shape)

# %% [markdown]
# ## Train Model
# 
# Training models without hyperparameter tuning for improvement comparision.

# %%
if torch.cuda.is_available():
    # Nvidia CUDA
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Apple Metal
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'Device: {device}')

# %% [markdown]
# ### Regressor Models

# %%
linear_model = LinearRegression()
svr_model = SVR()
rfr_model = RandomForestRegressor()
xgb_model = XGBRegressor()
lstm1_model = StockLSTM1()
lstm2_model = StockLSTM2()
gru1_model = StockGRU1()
gru2_model = StockGRU2()

models = {
    'LinearRegression': linear_model,
    'SVR': svr_model,
    'RFR': rfr_model,
    'XGB': xgb_model,
    'LSTM1': lstm1_model,
    'LSTM2': lstm2_model,
    'GRU1': gru1_model,
    'GRU2': gru2_model
}

# %%
results = dict()

for name, model in models.items():
    print(f"Training {name} ...")

    if (name == "SVR"):
        model.fit(norm_X_train, norm_y_train)
        y_pred_test = model.predict(norm_X_test)

    elif (name == "LSTM1") :
        model.load_state_dict(torch.load("./models/LSTM1.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM1_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)


    elif (name == "LSTM2") :
        model.load_state_dict(torch.load("./models/LSTM2.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM2_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    elif (name == "GRU1") :
        model.load_state_dict(torch.load("./models/GRU1.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU1_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    elif (name == "GRU2") :
        model.load_state_dict(torch.load("./models/GRU2.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU2_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    else:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
    
    mae = np.mean(np.abs(y_test.values - y_pred_test))
    mse = np.mean((y_test.values - y_pred_test) ** 2)
    rmse = np.sqrt(mse)
    direction = ( np.mean(np.sign(y_pred_test) == np.sign(y_test)) )

    results[name] = {
        'model': model,
        'y_pred_test': y_pred_test,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'direction': direction
    }

# %%
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Direction: {metrics['direction']:.6f}")

    plt.figure(figsize=(21, 9))
    # plt.plot(range(len(y_test)), y_test, label='Actual', marker='o')
    # plt.plot(range(len(y_test)), metrics['y_pred_test'], label='Predicted', marker='x')

    n_data = 200
    plt.plot(range(n_data), y_test[:n_data], label='Actual', marker='o')
    plt.plot(range(n_data), metrics['y_pred_test'][:n_data], label='Predicted', marker='x')
    plt.axhline(y=0)

    plt.xlabel('Date')
    plt.ylabel('target')
    plt.title(f'Actual vs Predicted ({name})')
    plt.legend()
    plt.show()

# %% [markdown]
# ### Classifier Models

# %%
logistic_model = LogisticRegression()
svc_model = SVC()
rfc_model = RandomForestClassifier()
xgbc_model = XGBClassifier()
knn_model = KNeighborsClassifier(n_neighbors=2)

classifier_models = {
    'LogisticRegression': logistic_model,
    'SVC': svc_model,
    'RFC': rfc_model,
    'XGBC': xgbc_model,
    'KNN': knn_model
}

# %%
classifier_results = dict()

for name, model in classifier_models.items():
    print(f"Training {name} ...")

    if (name == "SVC"):
        model.fit(norm_X_train_bool, norm_y_train_bool)
        y_pred_test = model.predict(norm_X_test_bool)

    else:
        model.fit(X_train_bool, y_train_bool)
        y_pred_test = model.predict(X_test_bool)

    accuracy = accuracy_score(y_test_bool, y_pred_test)
    report = classification_report(y_test_bool, y_pred_test)

    classifier_results[name] = {
        'model': model,
        'y_pred_test': y_pred_test,
        'accuracy': accuracy,
        'report': report
    }

# %%
for name, metrics in classifier_results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Report: {metrics['report']}")

    fpr, tpr, thresholds = roc_curve(y_test_bool, metrics['y_pred_test'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve ({name})")
    plt.legend(loc="lower right")
    plt.show()

# %% [markdown]
# # Hyper Parameter Tuning
# **Strategy**: Grid Search CV
# 
# Exhaustive search over specified parameter values for an estimator.

# %% [markdown]
# ### Regressor Models

# %%
tuned_linear_model = LinearRegression(fit_intercept=True)
tuned_svr_model = SVR(epsilon=0.01)
tuned_rfr_model = RandomForestRegressor(min_samples_leaf=9)
tuned_xgb_model = XGBRegressor(eta=0.1, max_depth=5)
tuned_lstm1_model = models["LSTM1"]
tuned_lstm2_model = models["LSTM2"]
tuned_gru1_model = models["GRU1"]
tuned_gru2_model = models["GRU2"]

tuned_models = {
    'tuned_LinearRegression': tuned_linear_model,
    'tuned_SVR': tuned_svr_model,
    'tuned_RFR': tuned_rfr_model,
    'tuned_XGB': tuned_xgb_model,
    'tuned_LSTM1' : tuned_lstm1_model,
    'tuned_LSTM2' : tuned_lstm2_model,
    'tuned_GRU1' : tuned_gru1_model,
    'tuned_GRU2' : tuned_gru2_model,
}

# %%
tuned_results = dict()

for name, model in tuned_models.items():
    print(f"Training {name} ...")

    if (name == "tuned_SVR"):
        model.fit(norm_X_train, norm_y_train * 10)
        y_pred_test = model.predict(norm_X_test) / 10

    elif (name == "tuned_LSTM1") :
        model.load_state_dict(torch.load("./models/LSTM1.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM1_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    elif (name == "tuned_LSTM2") :
        model.load_state_dict(torch.load("./models/LSTM2.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM2_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    elif (name == "tuned_GRU1") :
        model.load_state_dict(torch.load("./models/GRU1.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU1_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    elif (name == "tuned_GRU2") :
        model.load_state_dict(torch.load("./models/GRU2.pth.tar")['model'])

        model.to(device)
        model.eval()

        y_pred_all = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU2_stock_predict(model,ticker,X,y)

            y_pred_all = np.concatenate((y_pred_all,y_pred))

        y_pred_all = np.exp(y_pred_all) - 1
        _,y_pred_test = train_test_split(y_pred_all, random_state=42)

    else:
        model.fit(X_train, y_train * 10)
        y_pred_test = model.predict(X_test) / 10

    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    rmse = np.sqrt(mse)
    direction = ( np.mean(np.sign(y_pred_test) == np.sign(y_test)) )

    # incase this model needs to be saved
    # joblib.dump(model, f"./models/tuned/{name}.sav")

    tuned_results[name] = {
        'model': model,
        'y_pred_test': y_pred_test,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'direction': direction
    }

# %%
for name, metrics in tuned_results.items():
    print(f"Model: {name}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Direction: {metrics['direction']:.6f}")

    plt.figure(figsize=(21, 9))
    # plt.plot(range(len(y_test)), y_test, label='Actual', marker='o')
    # plt.plot(range(len(y_test)), metrics['y_pred_test'], label='Predicted', marker='x')

    n_data = 200
    plt.plot(range(n_data), y_test[:n_data], label='Actual', marker='o')
    plt.plot(range(n_data), metrics['y_pred_test'][:n_data], label='Predicted', marker='x')
    plt.axhline(y=0)

    plt.xlabel('Date')
    plt.ylabel('target')
    plt.title(f'Actual vs Predicted ({name})')
    plt.legend()
    plt.show()

# %% [markdown]
# ### Classifier Models

# %%
tuned_logistic_model = LogisticRegression(penalty='l2', C=0.01)
tuned_svc_model = SVC(degree=3, kernel='rbf')
tuned_rfc_model = RandomForestClassifier(min_samples_leaf=7)
tuned_xgbc_model = XGBClassifier(eta=0.2, subsample=0.9)
tuned_knn_model = KNeighborsClassifier(algorithm='auto', n_neighbors=2)

tuned_classifier_models = {
    'tuned_LogisticRegression': tuned_logistic_model,
    'tuned_SVC': tuned_svc_model,
    'tuned_RFC': tuned_rfc_model,
    'tuned_XGBC': tuned_xgbc_model,
    'tuned_KNN': tuned_knn_model
}

# %%
tuned_classifier_results = dict()

for name, model in tuned_classifier_models.items():
    print(f"Training {name} ...")

    if (name == 'tuned_SVC' or name == 'tuned_LogisticRegression'):
        model.fit(norm_X_train_bool, norm_y_train_bool)
        y_pred_test = model.predict(norm_X_test_bool)

    else:
        model.fit(X_train_bool, y_train_bool)
        y_pred_test = model.predict(X_test_bool)

    accuracy = accuracy_score(y_test_bool, y_pred_test)
    report = classification_report(y_test_bool, y_pred_test)

    tuned_classifier_results[name] = {
        'model': model,
        'y_pred_test': y_pred_test,
        'accuracy': accuracy,
        'report': report
    }

# %%
for name, metrics in tuned_classifier_results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Report: {metrics['report']}")

    fpr, tpr, thresholds = roc_curve(y_test_bool, metrics['y_pred_test'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve ({name})")
    plt.legend(loc="lower right")
    plt.show()

# %% [markdown]
# # Predicted Data

# %% [markdown]
# ### Regressor Models

# %%
predict_traintest = dict()
predict_evaluate = dict()

norm_evaluate_X = norm_evaluate[:, :-1]
norm_evaluate_y = norm_evaluate[-1]

evaluate_X = df_evaluate.drop('target', axis=1)
evaluate_y = df_evaluate.target

# Naive Forecast
print(f"Predicting NaiveForecast ...")
y_pred_traintest = data['Return_1d']
y_pred_evaluate = data['Return_1d']
predict_traintest['NaiveForecast'] = y_pred_traintest
predict_evaluate['NaiveForecast'] = y_pred_evaluate

for name, metrics in tuned_results.items():
    print(f"Predicting {name} ...")
    if (name == "tuned_SVR"):
        y_pred_traintest = metrics['model'].predict(norm_X) / 10
        y_pred_evaluate = metrics['model'].predict(norm_evaluate_X) / 10

    elif (name == "tuned_LSTM1") :
        metrics['model'].eval()

        y_pred_traintest = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM1_stock_predict(metrics['model'],ticker,X,y)

            y_pred_traintest = np.concatenate((y_pred_traintest,y_pred))
        y_pred_traintest = np.exp(y_pred_traintest) - 1

        y_pred_evaluate = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM1_stock_predict(metrics['model'],ticker,evaluate_X,evaluate_y)

            y_pred_evaluate = np.concatenate((y_pred_evaluate,y_pred))
        y_pred_evaluate = np.exp(y_pred_evaluate) - 1

    elif (name == "tuned_LSTM2") :
        metrics['model'].eval()

        y_pred_traintest = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM2_stock_predict(metrics['model'],ticker,X,y)

            y_pred_traintest = np.concatenate((y_pred_traintest,y_pred))
        y_pred_traintest = np.exp(y_pred_traintest) - 1

        y_pred_evaluate = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = LSTM2_stock_predict(metrics['model'],ticker,evaluate_X,evaluate_y)

            y_pred_evaluate = np.concatenate((y_pred_evaluate,y_pred))
        y_pred_evaluate = np.exp(y_pred_evaluate) - 1

    elif (name == "tuned_GRU1") :
        metrics['model'].eval()

        y_pred_traintest = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU1_stock_predict(metrics['model'],ticker,X,y)

            y_pred_traintest = np.concatenate((y_pred_traintest,y_pred))
        y_pred_traintest = np.exp(y_pred_traintest) - 1

        y_pred_evaluate = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU1_stock_predict(metrics['model'],ticker,evaluate_X,evaluate_y)

            y_pred_evaluate = np.concatenate((y_pred_evaluate,y_pred))
        y_pred_evaluate = np.exp(y_pred_evaluate) - 1

    elif (name == "tuned_GRU2") :
        metrics['model'].eval()

        y_pred_traintest = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU2_stock_predict(metrics['model'],ticker,X,y)

            y_pred_traintest = np.concatenate((y_pred_traintest,y_pred))
        y_pred_traintest = np.exp(y_pred_traintest) - 1

        y_pred_evaluate = np.array([], dtype=float)
        for ticker in data.index.unique('Ticker').to_list():
            y_pred,_ = GRU2_stock_predict(metrics['model'],ticker,evaluate_X,evaluate_y)

            y_pred_evaluate = np.concatenate((y_pred_evaluate,y_pred))
        y_pred_evaluate = np.exp(y_pred_evaluate) - 1

    else:
        y_pred_traintest = metrics['model'].predict(X) / 10
        y_pred_evaluate = metrics['model'].predict(evaluate_X) / 10

    predict_traintest[name] = y_pred_traintest
    predict_evaluate[name] = y_pred_evaluate

df_predict_traintest = pd.DataFrame(predict_traintest, index=df_traintest.index)
df_predict_evaluate = pd.DataFrame(predict_evaluate, index=df_evaluate.index)

# %% [markdown]
# ### Classifier Models

# %%
predict_traintest_bool = dict()
predict_evaluate_bool = dict()

norm_evaluate_X_bool = norm_evaluate[:, :-1]
norm_evaluate_y_bool = norm_evaluate[-1] > 0

evaluate_X_bool = df_evaluate.drop('target', axis=1)
evaluate_y_bool = df_evaluate.target > 0

# Naive Forecast
print(f"Predicting NaiveForecast ...")

for name, metrics in tuned_classifier_results.items():
    print(f"Predicting {name} ...")
    if (name == 'tuned_SVC' or name == 'tuned_LogisticRegression'):
        y_pred_traintest = metrics['model'].predict(norm_X_bool)
        y_pred_evaluate = metrics['model'].predict(norm_evaluate_X_bool)
    else:
        y_pred_traintest = metrics['model'].predict(X_bool)
        y_pred_evaluate = metrics['model'].predict(evaluate_X_bool)

    predict_traintest_bool[name] = y_pred_traintest
    predict_evaluate_bool[name] = y_pred_evaluate

df_predict_traintest_bool = pd.DataFrame(predict_traintest_bool, index=df_traintest.index)
df_predict_evaluate_bool = pd.DataFrame(predict_evaluate_bool, index=df_evaluate.index)

# %%
all_regression_models = { # model_name: model
    'NaiveForecast': None,
    'tuned_LinearRegression': tuned_linear_model,
    'tuned_SVR': tuned_svr_model,
    'tuned_RFR': tuned_rfr_model,
    'tuned_XGB': tuned_xgb_model
}

all_classifier_models = { # model_name: model
    'NaiveForecast': None,
    'tuned_LogisticRegression': tuned_logistic_model,
    'tuned_SVC': tuned_svc_model,
    'tuned_RFC': tuned_rfc_model,
    'tuned_XGBC': tuned_xgbc_model,
    'tuned_KNN': tuned_knn_model
}

df_predicted_reg = pd.concat([df_predict_traintest, df_predict_evaluate])
df_predicted_cls = pd.concat([df_predict_traintest_bool, df_predict_evaluate_bool]).astype(int)
df_predicted = pd.concat([df_predicted_reg, df_predicted_cls], axis=1)

# %% [markdown]
# # Trading

# %% [markdown]
# ## Simple Strategy
# - Position (Buy/Sell/do nothing) base on predicted '(log)return' of the next day
#     - return > 0 : buy
#     - return = 0 : do nothing
#     - return < 0 : sell
# - buy/sell all of portfolio in each transaction
# - no short position

# %% [markdown]
# ### Entire dataset

# %%
df_trading = None # init # for support duplicate run
df_trading = pd.DataFrame(data['target']).rename(columns={'target': 'Actual'})
df_trading = df_trading.join(df_predicted) # merge

# Trading return
df_trading_return = None # init
df_signal = ( df_predicted > 0 ).astype(int) # 1: Buy, 0: Sell
df_trading_return = df_signal.mul(df_trading['Actual'], axis=0)
df_trading_return['Buy&Hold'] = df_trading['Actual']
df_trading_return.columns = ['return_' + col for col in df_trading_return.columns]
df_trading = df_trading.join(df_trading_return) # merge

# Calculate cumulative return for each model
df_cumulative_return = None # init
df_cumulative_return = df_trading_return.groupby('Ticker').cumsum()
df_cumulative_return = df_cumulative_return.add(1)
df_cumulative_return.columns = ['cumulative_' + col for col in df_cumulative_return.columns]
df_trading = df_trading.join(df_cumulative_return)

# %%
df_cumulative_return.reset_index()

# %%
# Get the data for the latest day of each ticker
df_latest_day = df_cumulative_return.groupby(level='Ticker').tail(1)

# Reset the index to make 'Ticker' a column instead of an index
df_latest_day = df_latest_day.reset_index()

# Plot Cumulative return by model
for k in df_cumulative_return:
    plt.figure(figsize=(21, 9))
    sns.lineplot(
        data=df_cumulative_return.reset_index(),
        x="Date",
        y=k,
        hue="Ticker",
    )
    plt.legend(title="Ticker", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axvline(x=pd.Timestamp("2023-05-01"), color="slategray", linestyle="--")

    # Add label for the latest day with the latest cumulative return
    for i, row in df_latest_day.iterrows():
        plt.text(row['Date'], row[k], f'{row[k]*100:.0f}%', ha='left', va='bottom')

    plt.title(k)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    plt.show()


# %%
# Plot Cumulative return by ticker
for ticker in df_cumulative_return.index.get_level_values('Ticker').unique():
    plt.figure(figsize=(21, 9))
    sns.set_palette("Spectral",n_colors=len(df_cumulative_return.columns))
    for col in df_cumulative_return.columns:
        sns.lineplot(
            data=df_cumulative_return.loc[ticker].reset_index(),
            x="Date",
            y=col,
            label=col,
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axvline(x=pd.Timestamp("2023-05-01"), color="slategray", linestyle="--")
    plt.title(f'Cumulative Return for {ticker}')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.show()


# %%
# df_max_cumreturn = df_cumulative_return.groupby('Ticker').max() * 100  # percent
# df_min_cumreturn = df_cumulative_return.groupby('Ticker').min() * 100  # percent
# df_maxdrawdown = df_min_cumreturn - df_max_cumreturn

# # Plot maximum drawdown
# plt.figure(figsize=(12, 6))
# df_maxdrawdown.plot(kind='bar', ax=plt.gca())
# plt.xlabel('Ticker')
# plt.ylabel('Max Drawdown (%)')
# plt.title('Maximum Drawdown for Different Tickers')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Plot per ticker maximum drawdown
# for ticker in df_maxdrawdown.index:
#     plt.figure(figsize=(12, 6))
#     ax = sns.barplot(x=df_maxdrawdown.columns, y=df_maxdrawdown.loc[ticker], palette="rocket_r", order=df_maxdrawdown.loc[ticker].sort_values(ascending=False).index)
#     plt.xlabel('Model')
#     plt.ylabel('Max Drawdown (%)')
#     plt.title(f'Maximum Drawdown for {ticker}')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     # Annotate each bar with its value
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, -10), textcoords='offset points')
#     plt.show()

# %% [markdown]
# ### Evaluation set

# %%
# Evaluation time horizon
start_date = pd.Timestamp("2023-05-01")
end_date = pd.Timestamp("2024-04-30")

# Calculate cumulative return for each model in validation set
df_trading_return_eval = df_trading_return.reset_index()
df_trading_return_eval = df_trading_return_eval[
    (df_trading_return_eval["Date"] >= start_date)
    & (df_trading_return_eval["Date"] <= end_date)
]
df_trading_return_eval = df_trading_return_eval.set_index(['Ticker', 'Date'])

df_cumulative_return_eval = None # init
df_cumulative_return_eval = df_trading_return_eval.groupby('Ticker').cumsum()
df_cumulative_return_eval = df_cumulative_return_eval.add(1)
df_cumulative_return_eval.columns = ['cumulative_' + col for col in df_cumulative_return_eval.columns]

# Plot Cumulative return by model
for k in df_cumulative_return:
    plt.figure(figsize=(21, 9))
    sns.lineplot(
        data=df_cumulative_return_eval.reset_index(),
        x="Date",
        y=k,
        hue="Ticker",
    )
    plt.legend(title="Ticker", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axvline(x=pd.Timestamp("2023-05-01"), color="slategray", linestyle="--")

    plt.title(f'{k} on Evaluation set')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    plt.show()

# %%
# Plot Cumulative return by ticker
for ticker in df_cumulative_return_eval.index.get_level_values('Ticker').unique():
    plt.figure(figsize=(21, 9))
    sns.set_palette("Spectral",n_colors=len(df_cumulative_return.columns))
    for col in df_cumulative_return_eval.columns:
        sns.lineplot(
            data=df_cumulative_return_eval.loc[ticker].reset_index(),
            x="Date",
            y=col,
            label=col,
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axvline(x=pd.Timestamp("2023-05-01"), color="slategray", linestyle="--")
    plt.title(f'Cumulative Return for {ticker} on Eval set')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.show()

# %%
# df_max_cumreturn_eval = df_cumulative_return_eval.groupby('Ticker').max() * 100  # percent
# df_min_cumreturn_eval = df_cumulative_return_eval.groupby('Ticker').min() * 100  # percent
# df_maxdrawdown_eval = df_min_cumreturn_eval - df_max_cumreturn_eval

# # Plot All maximum drawdown
# plt.figure(figsize=(21, 9))
# df_maxdrawdown_eval.plot(kind='bar', ax=plt.gca())
# plt.xlabel('Ticker')
# plt.ylabel('Max Drawdown (%)')
# plt.title('Maximum Drawdown for Different Tickers on Eval set')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Plot per ticker maximum drawdown
# for ticker in df_maxdrawdown_eval.index:
#     plt.figure(figsize=(12, 6))
#     ax = sns.barplot(x=df_maxdrawdown_eval.columns, y=df_maxdrawdown_eval.loc[ticker], palette="rocket_r", order=df_maxdrawdown_eval.loc[ticker].sort_values(ascending=False).index)
#     plt.xlabel('Model')
#     plt.ylabel('Max Drawdown (%)')
#     plt.title(f'Maximum Drawdown for {ticker} on Eval set')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     # Annotate each bar with its value
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, -10), textcoords='offset points')
#     plt.show()

# %% [markdown]
# ## 1/N Portfolio

# %%
# Entire set
df_cumulative_return_eq = df_cumulative_return.groupby('Date').mean()

plt.figure(figsize=(21, 9))
for col in df_cumulative_return_eq.columns:
    sns.set_palette("Spectral",n_colors=len(df_cumulative_return.columns))
    sns.lineplot(
        data=df_cumulative_return_eq.reset_index(),
        x="Date",
        y=col,
        label=col,
    )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.axvline(x=pd.Timestamp("2023-05-01"), color="slategray", linestyle="--")
plt.title(f'Cumulative Return Equal Weight')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()


# %%
# Evaluation set
df_cumulative_return_eq_eval = df_cumulative_return_eval.groupby('Date').mean()

plt.figure(figsize=(21, 9))
for col in df_cumulative_return_eq_eval.columns:
    sns.set_palette("Spectral",n_colors=len(df_cumulative_return.columns))
    sns.lineplot(
        data=df_cumulative_return_eq_eval.reset_index(),
        x="Date",
        y=col,
        label=col,
    )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.axvline(x=pd.Timestamp("2023-05-01"), color="slategray", linestyle="--")
plt.title(f'Cumulative Return Equal Weight on Eval set')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()


# %% [markdown]
# ### Sharpe ratio
# - Usually, any Sharpe ratio greater than 1.0 is considered acceptable to good by investors.
# - A ratio higher than 2.0 is rated as very good.
# - A ratio of 3.0 or higher is considered excellent.
# - A ratio under 1.0 is considered sub-optimal.

# %%
# Entire set # 10y
# Calculate average daily return and standard deviation
average_daily_return = (df_cumulative_return_eq - 1).mean()
std_dev_daily_return = (df_cumulative_return_eq - 1).std()

# https://fred.stlouisfed.org/series/TB3MS
risk_free_rate = 0.03 / 252

# Calculate Sharpe Ratio
sharpe_ratio = (average_daily_return - risk_free_rate) / std_dev_daily_return

# Print Sharpe Ratio
print("Sharpe Ratio:")
print(sharpe_ratio)


# %%
sharpe_ratio-sharpe_ratio['cumulative_return_Buy&Hold']

# %%
# Eval set # 1y
# Calculate average daily return and standard deviation
average_daily_return = (df_cumulative_return_eq_eval - 1).mean()
std_dev_daily_return = (df_cumulative_return_eq_eval - 1).std()

# https://fred.stlouisfed.org/series/TB3MS
risk_free_rate = 0.03 / 252

# Calculate Sharpe Ratio
sharpe_ratio = (average_daily_return - risk_free_rate) / std_dev_daily_return

# Print Sharpe Ratio
print("Sharpe Ratio:")
print(sharpe_ratio)


# %%
sharpe_ratio-sharpe_ratio['cumulative_return_Buy&Hold']

# %% [markdown]
# # FeatureImportance

# %%

def get_feature_importance(model, feature_names=None):
    if hasattr(model, 'coef_'):
        feature_importance = {feature_names[i]: coef for i, coef in enumerate(model.coef_)}
        df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
        df = df.abs()
        df = df.sort_values(by='Importance', ascending=False)
        return df
    
    else:
        print("Warning: Model doesn't have coef_ attribute. Feature importance cannot be extracted.")
        return None

# %%
print(tuned_results.keys())
print(tuned_classifier_results.keys())

# %%
feature_names = ["Dollar_Volume","RSI","BB_High","BB_Mid","BB_Low","ATR","NATR","MACD","Return_1d","Return_5d","Return_10d","Return_21d","Return_42d","Return_63d","High_Return","Low_Return","Open_Return","Momentum_5","Momentum_10","Momentum_5_10","Momentum_21","Momentum_5_21","Momentum_42","Momentum_5_42","Momentum_63","Momentum_5_63","Year","Month"]
regression_feature_importance_df = get_feature_importance(tuned_results["tuned_LinearRegression"]["model"], feature_names)
regression_feature_importance_df

# %%
svc_feature_importance_df = FeatureImportance.get_svc_feature_importance(tuned_classifier_results["tuned_SVC"]["model"], feature_names)

# %%
permutation_importance_df = FeatureImportance.get_svr_feature_importance(tuned_results["tuned_SVR"]["model"], X, y, feature_names=feature_names)

# %%
xgboost_feature_importance_df = FeatureImportance.get_xgboost_feature_importance(tuned_results['tuned_XGB']["model"], feature_names)
xgboost_feature_importance_df

# %%
rfr_feature_importance_df = FeatureImportance.get_rf_feature_importance(tuned_results['tuned_RFR']["model"], feature_names)
rfr_feature_importance_df


