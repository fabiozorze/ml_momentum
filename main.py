import pandas as pd
import numpy as np
from ML import Feature_eng_modules as eg
from ML import Modules as fm
from ML import model_run as mr
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import shap
import yfinance as yf

import matplotlib
matplotlib.use('TkAgg')

stocks = []

dic_preds = {}

for stock in stocks:

    data = yf.download(stock)
    data = data[['Open','High','Low','Close','Volume']]
    data.columns = ['open','high','low','close','volume']

    windows = [2,3,5,9,15,30]
    features = eg.add_and_save_features_ALL(data, windows)

    df_normalized = (features - features.mean()) / features.std()

    binary_outcomes = np.sign(df_normalized['close'].pct_change().shift(-5))

    df_normalized = df_normalized.iloc[-150:]
    binary_outcomes = binary_outcomes.iloc[-150:]

    pred = mr.run_model(df_normalized,binary_outcomes,ohcl_data=data,LKBK=100,LFWD=5,shap_th=0.025)

    dic_preds[stock] = pred


