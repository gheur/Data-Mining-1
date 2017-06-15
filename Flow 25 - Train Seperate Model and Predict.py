import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator, H2ODeepLearningEstimator


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
n_iterations = 10
pd_train = pd.read_csv('fe_train.csv')
pd_test = pd.read_csv('fe_test.csv')

sj_train =  pd_train[pd_train['city'] == 'sj']
iq_train =  pd_train[pd_train['city'] == 'iq']
sj_test =  pd_test[pd_test['city'] == 'sj']
iq_test = pd_test[pd_test['city'] == 'iq']
pd_submit = pd.read_csv('dataset/submission_format.csv')

total_cases_sj = list(sj_train['total_cases'])
total_cases_iq = list(iq_train['total_cases'])

# Make connection to h2o server
h2o.init(nthreads=-1, max_mem_size_GB=8)
h2o.remove_all()

# Create h2o frames
sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))
sj_testing_frame = h2o.H2OFrame(sj_test)
sj_testing_frame.set_names(list(sj_test.columns))
iq_testing_frame = h2o.H2OFrame(iq_test)
iq_testing_frame.set_names(list(iq_test.columns))

col_names = list(sj_testing_frame.names)
for col in col_names:
    train_col_type = str(sj_training_frame.type(col))
    test_col_type = str(sj_testing_frame.type(col))
    if test_col_type != train_col_type:
        if train_col_type == 'real':
            sj_testing_frame[col] = sj_testing_frame[col].asnumeric()
        elif train_col_type == 'enum':
            sj_testing_frame[col] = sj_testing_frame[col].asfactor()

col_names = list(iq_testing_frame.names)
for col in col_names:
    train_col_type = str(iq_training_frame.type(col))
    test_col_type = str(iq_testing_frame.type(col))
    if test_col_type != train_col_type:
        if train_col_type == 'real':
            iq_testing_frame[col] = iq_testing_frame[col].asnumeric()
        elif train_col_type == 'enum':
            iq_testing_frame[col] = iq_testing_frame[col].asfactor()

# Training parameters
input_columns = list(sj_train.columns)
response_column = 'total_cases'
input_columns.remove(response_column)

# Models
sj_min_mae = 1000
sj_best_model = None

iq_min_mae = 1000
iq_best_model = None

for i in range(n_iterations):
    model_sj = H2ODeepLearningEstimator(nfolds=10, hidden=[512, 512])
    model_sj.train(x=input_columns, y=response_column, training_frame=sj_training_frame)

    model_iq = H2ODeepLearningEstimator(nfolds=10, hidden=[512, 512])
    model_iq.train(x=input_columns, y=response_column, training_frame=iq_training_frame)

    if model_sj.mae() < sj_min_mae:
        sj_min_mae = model_sj.mae()
        sj_best_model = model_sj

    if model_iq.mae() < iq_min_mae:
        iq_min_mae = model_iq.mae()
        iq_best_model = model_iq

print("Model SJ MAE  :", sj_min_mae)
# h2o.save_model(model=sj_best_model, path='sj_model', force=True)

print("Model IQ MAE  :", iq_min_mae)
# h2o.save_model(model=iq_best_model, path='iq_model', force=True)

prediction_sj = sj_best_model.predict(test_data=sj_testing_frame)
prediction_iq = iq_best_model.predict(test_data=iq_testing_frame)

prediction_sj = h2OColumnToList(prediction_sj, data_type='real')
prediction_iq = h2OColumnToList(prediction_iq, data_type='real')

predictions = prediction_sj
predictions.extend(prediction_iq)

for i in range(len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = 1
    else:
        predictions[i] = int(predictions[i] + 0.5)

pd_submit['total_cases'] = pd.Series(predictions)
pd_submit.to_csv("submit.csv", index=False)











