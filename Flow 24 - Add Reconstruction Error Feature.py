import pandas as pd
import h2o
import Dataset
from h2o.estimators import H2OAutoEncoderEstimator


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


pd_train = Dataset.PD_TRAIN

sj_train = pd_train[pd_train['city'] == 'sj']
iq_train = pd_train[pd_train['city'] == 'iq']

del sj_train['city']
del iq_train['city']

# input columns
input_columns = list(pd_train.columns)
input_columns.remove('city')
input_columns.remove('total_cases')

# Initialize h2o server
h2o.init()

# Create h2o frames
sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))

# Define model
sj_model = H2OAutoEncoderEstimator()
iq_model = H2OAutoEncoderEstimator()

# Train models
sj_model.train(x=input_columns, training_frame=sj_training_frame)
iq_model.train(x=input_columns, training_frame=iq_training_frame)

# Get reconstruction error
sj_reconstruction_error = sj_model.anomaly(test_data=sj_training_frame, per_feature=False)
iq_reconstruction_error = iq_model.anomaly(test_data=iq_training_frame, per_feature=False)

sj_reconstruction_error = list(map(float, h2OColumnToList(sj_reconstruction_error)))
iq_reconstruction_error = list(map(float, h2OColumnToList(iq_reconstruction_error)))

recon_error = []
recon_error.extend(sj_reconstruction_error)
recon_error.extend(iq_reconstruction_error)

pd_train['reconstruction_error'] = recon_error
pd_train.to_csv('fe_train.csv', index=False)


# Processing Testing Frame
pd_train = Dataset.PD_TEST

sj_train = pd_train[pd_train['city'] == 'sj']
iq_train = pd_train[pd_train['city'] == 'iq']

del sj_train['city']
del iq_train['city']

# input columns
input_columns = list(pd_train.columns)
input_columns.remove('city')

# Create h2o frames
sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))

# Define model
sj_model = H2OAutoEncoderEstimator()
iq_model = H2OAutoEncoderEstimator()

# Train models
sj_model.train(x=input_columns, training_frame=sj_training_frame)
iq_model.train(x=input_columns, training_frame=iq_training_frame)

# Get reconstruction error
sj_reconstruction_error = sj_model.anomaly(test_data=sj_training_frame, per_feature=False)
iq_reconstruction_error = iq_model.anomaly(test_data=iq_training_frame, per_feature=False)

sj_reconstruction_error = list(map(float, h2OColumnToList(sj_reconstruction_error)))
iq_reconstruction_error = list(map(float, h2OColumnToList(iq_reconstruction_error)))

recon_error = []
recon_error.extend(sj_reconstruction_error)
recon_error.extend(iq_reconstruction_error)

pd_train['reconstruction_error'] = recon_error
pd_train.to_csv('fe_test.csv', index=False)

