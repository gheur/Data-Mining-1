import pandas as pd
import h2o
from h2o.estimators import H2OAutoEncoderEstimator, H2OGradientBoostingEstimator

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

h2o.init()
pd_train = pd.read_csv('na_filled_random_forest.csv')
training_frame = h2o.H2OFrame(pd_train)
columns = list(pd_train.columns)
anomaly_model = H2OAutoEncoderEstimator()
anomaly_model.train(x=columns, training_frame=training_frame)
reconstruction_error = anomaly_model.anomaly(test_data=training_frame, per_feature=False)
reconstruction_error = list(map(float, h2OColumnToList(reconstruction_error)))
pd_train['reconstruction_error'] = reconstruction_error

pd_test = pd.read_csv('dataset/dengue_features_test.csv')
testing_frame = h2o.H2OFrame(pd_test)
columns = list(pd_test.columns)
anomaly_model = H2OAutoEncoderEstimator()
anomaly_model.train(x=columns, training_frame=testing_frame)
reconstruction_error = anomaly_model.anomaly(test_data=testing_frame, per_feature=False)
reconstruction_error = list(map(float, h2OColumnToList(reconstruction_error)))
pd_test['reconstruction_error'] = reconstruction_error


training_frame = h2o.H2OFrame(pd_train)
training_frame.set_name(list(pd_train.columns))
testing_frame = h2o.H2OFrame(pd_test)
testing_frame.set_name(list(pd_test.columns))
training_columns = list(pd_train.columns)
training_columns.remove('total_cases')
response_column = 'total_cases'
model = H2OGradientBoostingEstimator(nfolds=5, ntrees=100, max_depth=20, balance_classes=True)
model.train(x=training_columns, y=response_column, training_frame=training_frame);
predictions = model.predict(test_data=testing_frame)
predictions = h2OColumnToList(predictions, data_type='real')
pd_submit = pd.read_csv('dataset/submission_format.csv')
pd_submit['total_cases'] = pd.Series(predictions)
pd_submit.to_csv("submit.csv", index=False)



