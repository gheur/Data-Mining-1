# Filling missing values with machine learning approach : Success
# https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/

import pandas as pd
import numpy as np
import math


def getNALocation(data_frame, column_name):
    """
    Identify locations of missing values

    :param data_frame: pandas data frame    [pd.DataFrame]
    :param column_name: name of the column  [str]
    :return: list of indices                [list]
    """

    # Validating parameters
    if not isinstance(data_frame, pd.DataFrame) or not isinstance(column_name, str):
        return

    na_indices = data_frame[column_name].index[data_frame[column_name].apply(np.isnan)]
    indices = data_frame.index.values.tolist()
    return [indices.index(i) for i in na_indices]


def insertValuesToFrame(data_frame, column_name, index, values):
    """
    Insert values to column in a frame at given indices
    :param data_frame: input data frame where data need to be added [pd.DataFrame]
    :param column_name: target column name                          [str]
    :param index: list of indexes                                   [int list]
    :param values: input values                                     [str/number]
    :return: data frame with inserted values                        [pd.DataFrame]
    """

    if(not isinstance(data_frame, pd.DataFrame)
       or not isinstance(column_name, str)
       or not isinstance(index, list)
       or not isinstance(values, list)
       or len(index) != len(values)):
        return

    column_values = list(data_frame[column_name])
    for i in range(len(index)):
        column_values[index[i]] = values[i]

    column_values = pd.Series(column_values)
    data_frame[column_name] = column_values

    return data_frame


def generateMissingValues(missing_index, data_column, n_rows, averaging_factor):
    """
    Generate new values to missing indexes [Average]
    :param n_rows: Number of rows in data
    :param missing_index: Missing value locations
    :param data_column: Original data column
    :return: Completed column data
    """

    left = int(averaging_factor / 2)
    generated_values = []
    for i in missing_index:
        index = i - left
        if index < 0:
            index = 0

        count = 0
        total = 0.0
        while index < n_rows or count < averaging_factor:
            if not math.isnan(data_column[index]):
                total += data_column[index]
                count += 1
            index += 1
        averaged_value = total / count
        generated_values.append(averaged_value)

    for i in range(len(missing_index)):
        data_column[missing_index[i]] = generated_values[i]

    return data_column


pd_train = pd.read_csv('dataset/dengue_features_train.csv')
pd_labels = pd.read_csv('dataset/dengue_labels_train.csv')
na_columns = pd_train.columns[pd_train.isnull().any()]
n_rows = len(pd_train)

for column_name in na_columns:
    head = "Generate missing values for: " + column_name
    separate = "-" * len(head)
    print(head)
    print(separate)

    # Identifying missing value locations
    na_indices = getNALocation(pd_train, column_name)
    generated_values = generateMissingValues(missing_index=na_indices, data_column=list(pd_train[column_name]), n_rows=n_rows, averaging_factor=5)
    pd_train[column_name] = generated_values

pd_train['total_cases'] = pd_labels['total_cases']
pd_train.to_csv(path_or_buf='na_filled_averaged.csv', index=False)
