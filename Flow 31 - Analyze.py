import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def NormalizeScale(data_frame, column_names):
    """
    Normalize selected columns
    :param data_frame: input data frame         [pandas.DataFrame]
    :param column_names: selected column names  [list]
    :return: normalized frame                   [pandas.DataFrame]
    """

    for column in column_names:
        data_frame[column] = (data_frame[column] - data_frame[column].mean()) / data_frame[column].std()

    return data_frame


def MinMaxScale(data_frame, column_names):
    """
    Min Max scale selected columns
    :param data_frame: input data frame                     [pandas.DataFrame]
    :param min_value: minimum value of the given series     [int]
    :param max_value: maximum value of the given series     [int]
    :return: min max scaled frame
    """

    for column in column_names:
        data_frame[column] = (data_frame[column] - data_frame[column].min()) / \
                             (data_frame[column].max() - data_frame[column].min())

    return data_frame


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]



# SJ train
column = 'precipitation_amt_mm'

sj_train = pd.read_csv('sj_train_original.csv')
print list(sj_train.columns)

sj_indexes = list(sj_train.index)
sj_precipitation = pd.Series(sj_train[column]).rolling(window=5).std()
sj_total_cases = sj_train['total_cases']


plt.plot(sj_indexes, sj_precipitation)
plt.plot(sj_indexes, sj_total_cases)
plt.title(column + ' vs total cases')
plt.legend([column, 'total_cases'], loc='upper right')
plt.show()

# IQ train
iq_train = pd.read_csv('iq_train_original.csv')

iq_indexes = list(iq_train.index)
iq_precipitation = pd.Series(iq_train[column]).rolling(window=5).mean()
iq_total_cases = iq_train['total_cases']


plt.plot(iq_indexes, iq_precipitation)
plt.plot(iq_indexes, iq_total_cases)
plt.title(column + ' vs total cases')
plt.legend([column, 'total_cases'], loc='upper right')
plt.show()
