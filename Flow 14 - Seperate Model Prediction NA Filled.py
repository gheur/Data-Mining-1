import h2o
import Dataset
import pandas as pd


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


# Initialize server
h2o.init()

# Load models
model_sj = h2o.load_model('sj_model/GBM_model_python_1495247918841_1')
model_iq = h2o.load_model('iq_model/GBM_model_python_1495247918841_2')

# Load testing data
pd_testing = Dataset.PD_TEST
pd_submit = Dataset.PD_SUBMIT

sj_test = pd_testing[pd_testing['city'] == 'sj']
iq_test = pd_testing[pd_testing['city'] == 'iq']

del sj_test['city']
del iq_test['city']

# Create h2o frames
sj_test_frame = h2o.H2OFrame(sj_test)
sj_test_frame.set_names(list(sj_test.columns))
iq_test_frame = h2o.H2OFrame(iq_test)
iq_test_frame.set_names(list(iq_test.columns))

prediction_sj = model_sj.predict(test_data=sj_test_frame)
prediction_iq = model_iq.predict(test_data=iq_test_frame)

prediction_sj = h2OColumnToList(prediction_sj, data_type='real')
prediction_iq = h2OColumnToList(prediction_iq, data_type='real')

predictions = prediction_sj
predictions.extend(prediction_iq)

for i in range(len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = 1
    else:
        predictions[i] = int(predictions[i] + 0.5)

pd_submit[Dataset.RESPONSE_COLUMN] = pd.Series(predictions)
pd_submit.to_csv("submit.csv", index=False)





