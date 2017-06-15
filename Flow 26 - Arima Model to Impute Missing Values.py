# http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
import numpy as np
import Dataset
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

# Configuration
max_p = 5

# Load train data set
pd_train = pd.read_csv('fe_train.csv')
na_columns = list(pd_train.columns[pd_train.isnull().any()])
na_columns.remove('ndvi_ne')
na_columns.remove('ndvi_nw')
na_columns.remove('ndvi_se')
na_columns.remove('ndvi_sw')

print("Start imputing training frame")
for column in na_columns:
    print("Column name              : " + column)

    train_series = list(pd_train[column])
    nan_indices = []
    for i in range(len(train_series)):
        if np.isnan(train_series[i]) or train_series[i] == 0.0:
            nan_indices.append(i)

    print("Number of missing values : " + str(len(nan_indices)))
    print("-------------------------------------")

    for nan_id in nan_indices:
        for i in range(max_p, 0, -1):
            try:
                sub_series = train_series[:nan_id]
                model = ARIMA(sub_series, order=(i,1,0))
                model_fit = model.fit(disp=0)
                predicted_value = model_fit.forecast()[0][0]
                print(nan_id, predicted_value)
                train_series[nan_id] = predicted_value
                break
            except ValueError:
                print("Increasing degree of freedom")
                continue
    pd_train[column] = train_series

    print("-------------------------------------")

print("Saving fe_train_imputed.csv")
pd_train.to_csv('fe_train_imputed.csv', index=False)

# Load test data set
pd_test = pd.read_csv('fe_test.csv')
na_columns = list(pd_test.columns[pd_test.isnull().any()])
na_columns.remove('ndvi_ne')
na_columns.remove('ndvi_nw')
na_columns.remove('ndvi_se')
na_columns.remove('ndvi_sw')

print("Start imputing testing frame")
for column in na_columns:
    print("Column name : " + column)

    test_series = list(pd_test[column])
    nan_indices = []
    for i in range(len(test_series)):
        if np.isnan(test_series[i]) or test_series[i] == 0.0:
            nan_indices.append(i)

    print("Number of missing values : " + str(len(nan_indices)))
    print("-------------------------------------")

    for nan_id in nan_indices:
        for i in range(max_p, 0, -1):
            try:
                sub_series = test_series[:nan_id]
                model = ARIMA(sub_series, order=(i, 1, 0))
                model_fit = model.fit(disp=0)
                predicted_value = model_fit.forecast()[0][0]
                print(nan_id, predicted_value)
                test_series[nan_id] = predicted_value
                break
            except ValueError:
                print("Increasing degree of freedom")
                continue
    pd_test[column] = test_series
    print("-------------------------------------")

print("Saving fe_test_imputed.csv")
pd_test.to_csv('fe_test_imputed.csv', index=False)


