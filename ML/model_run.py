import pandas as pd
import numpy as np
from ML import Feature_eng_modules as eg
from ML import Modules as fm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import shap
import yfinance as yf


def run_model(df_normalized,binary_outcomes,ohcl_data=None,LKBK=None,LFWD=None,shap_th=None):

    ml_data = pd.concat([df_normalized,binary_outcomes],axis=1).fillna(method='ffill').dropna()
    features = ml_data.iloc[::,:-1]
    binary_outcomes = ml_data.iloc[::,-1:]

    prediction_data = pd.DataFrame()
    index_list = []

    fea_se = {}

    for i in range(LKBK,len(features),LFWD):

        print(i)

        X_train_data = features[i - LKBK:i]
        Y_train_data = binary_outcomes[i - LKBK:i]

        Y_train_data = np.where(Y_train_data == 0,-1,Y_train_data)

        ### Scale
        scaler = StandardScaler()
        X_train_data_scaled = scaler.fit_transform(X_train_data)

        ### Model Train
        rf_clf = fm.run_rf_model_2(X_train_data_scaled, Y_train_data, params=None)

        ### Calculate SHAP values
        X_train_data_scaled_df = pd.DataFrame(X_train_data_scaled, columns=X_train_data.columns)
        explainer = shap.TreeExplainer(rf_clf)
        shap_values = explainer.shap_values(X_train_data_scaled_df)
        mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

        # Select features with mean absolute SHAP values higher than 0.25
        selected_features = np.where(mean_abs_shap_values > shap_th)

        features_selected = X_train_data_scaled_df.columns[np.unique(selected_features[1])]
        fea_se[i] = features_selected

        position = np.unique(selected_features[1])

        scaler = StandardScaler()
        X_train_data_scaled = scaler.fit_transform(X_train_data.iloc[:, position])

        rf_clf = fm.run_rf_model_2(X_train_data_scaled, Y_train_data, params=None)

        ### Model Test
        index_data = features.index[i:i+LFWD]
        X_test_data = pd.DataFrame(features[i:i+LFWD])

        X_test_data = scaler.transform(X_test_data.iloc[:, position])
        pred = pd.DataFrame(rf_clf.predict_proba(X_test_data))

        index_list.append(index_data)
        prediction_data = pd.concat([prediction_data, pred])


    close = ohcl_data['close'][LKBK:]

    prediction_data.index = features.iloc[LKBK:].index

    prediction_data['close'] = close

    return prediction_data




