# Filling missing values with machine learning approach : Success
# https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/
import h2o
import pandas as pd
import numpy as np
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator


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


def h2OColumnToList(data_column, data_type="enum"):
    """
    Parsed values in a given h2o column to a python list with given data type

    :param data_column: h2o data frame                      [h2o.H2OFrame]
    :param data_type: data type of the column (real/enum)   [str]
    :return: parsed values                                  [list]
    """

    # Validating parameters
    if not isinstance(data_column, h2o.H2OFrame) or not isinstance(data_type, str):
        return

    value_str = data_column.get_frame_data()
    splitter_list = value_str.split("\n")[1:-1]

    if data_type == "real":
        return list(map(float, splitter_list))
    elif data_type == "enum":
        return splitter_list


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


h2o.init(nthreads=-1)

pd_train = pd.read_csv('dataset/dengue_features_train.csv')
pd_labels = pd.read_csv('dataset/dengue_labels_train.csv')
na_columns = pd_train.columns[pd_train.isnull().any()]

h_train = h2o.H2OFrame(pd_train)
h_train.set_names(list(pd_train.columns))

for response_column in na_columns:
    head = "Generated missing values for: " + response_column
    separate = "-" * len(head)
    print(head)

    # Setting up training columns
    training_columns = list(h_train.names)
    training_columns.remove(response_column)

    # Identifying missing value locations
    na_indices = getNALocation(pd_train, response_column)
    print("Missing", na_indices)

    # Data set with missing values
    pd_test = pd_train.ix[na_indices]
    del pd_test[response_column]
    h_test = h2o.H2OFrame(pd_test)
    h_test.set_names(list(pd_test.columns))

    # Check for type mismatches in train and test data
    col_names = list(h_test.names)
    for col in col_names:
        train_col_type = str(h_train.type(col))
        test_col_type = str(h_test.type(col))
        if test_col_type != train_col_type:
            if train_col_type == 'real':
                h_test[col] = h_test[col].asnumeric()
            elif train_col_type == 'enum':
                h_test[col] = h_test[col].asfactor()

    # Define machine learning model to predict missing values
    model = H2ORandomForestEstimator()
    model.train(x=training_columns, y=response_column, training_frame=h_train)

    prediction = model.predict(test_data=h_test)

    value_list = h2OColumnToList(prediction, str(h_train.type(col=response_column)))
    pd_train = insertValuesToFrame(data_frame=pd_train, column_name=response_column, index=na_indices, values=value_list)

    # Display result
    head = "Generated missing values for: " + response_column
    separate = "-" * len(head)
    print(head)
    print(separate)
    print(response_column + " : #Missing values " + str(len(na_indices)))
    print("\n\n")

pd_train['total_cases'] = pd_labels['total_cases']
pd_train.to_csv(path_or_buf='na_filled_random_forest.csv', index=False)
