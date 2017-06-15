import pandas as pd
import h2o
from h2o.estimators import H2OAutoEncoderEstimator, H2OGradientBoostingEstimator, H2ODeepLearningEstimator


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


sj_train = pd.read_csv('sj_shift_train.csv')
iq_train = pd.read_csv('iq_shift_train.csv')
sj_test = pd.read_csv('sj_test_original.csv')
iq_test = pd.read_csv('iq_train_original.csv')
pd_submit = pd.read_csv('dataset/submission_format.csv')

columns = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
           'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
           'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
           'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
           'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c',
           'station_precip_mm']

# # Drop columns in sj less than 0.1
# dropping_threshold_sj = 0.2
# print "SJ Train : Dropping columns"
# print "---------------------------"
# print "Threshold : " + str(dropping_threshold_sj)
# for column in columns:
#     correlation = sj_train['total_cases'].corr(sj_train[column])
#     if correlation < dropping_threshold_sj:
#         print column
#         del sj_train[column]
# print "---------------------------"
# print ""
#
# # Drop columns in iq less than 0.18
# dropping_threshold_iq = 0.08
# print "IQ Train : Dropping columns"
# print "---------------------------"
# print "Threshold : " + str(dropping_threshold_iq)
# for column in columns:
#     correlation = iq_train['total_cases'].corr(iq_train[column])
#     if correlation < dropping_threshold_iq:
#         print column
#         del iq_train[column]
# print "---------------------------"
# print ""

print "SJ Features : " + str(len(list(sj_train.columns)))
print "--------------------"
print list(sj_train.columns)
print ""
print "IQ Features : " + str(len(list(iq_train.columns)))
print "--------------------"
print list(iq_train.columns)
print ""

h2o.init()

print "Adding Reconstruction Error"
print "---------------------------"
print "Applying to SJ Train"
print "---------------------------"
columns = list(sj_train.columns)
columns.remove('total_cases')
sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
sj_testing_frame = h2o.H2OFrame(sj_test)
sj_testing_frame.set_names(list(sj_test.columns))
sj_model = H2OAutoEncoderEstimator()
sj_model.train(x=columns, training_frame=sj_training_frame)
sj_reconstruction_error = sj_model.anomaly(test_data=sj_training_frame, per_feature=False)
sj_reconstruction_error = list(map(float, h2OColumnToList(sj_reconstruction_error)))
sj_reconstruction_error_test = sj_model.anomaly(test_data=sj_testing_frame, per_feature=False)
sj_reconstruction_error_test = list(map(float, h2OColumnToList(sj_reconstruction_error_test)))
sj_train['reconstruction_error'] = sj_reconstruction_error
sj_test['reconstruction_error'] = sj_reconstruction_error_test

print ""

print "Applying to IQ Train"
print "---------------------------"
columns = list(iq_train.columns)
columns.remove('total_cases')
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))
iq_testing_frame = h2o.H2OFrame(iq_test)
iq_testing_frame.set_names(list(iq_test.columns))
iq_model = H2OAutoEncoderEstimator()
iq_model.train(x=columns, training_frame=iq_training_frame)
iq_reconstruction_error = iq_model.anomaly(test_data=iq_training_frame, per_feature=False)
iq_reconstruction_error = list(map(float, h2OColumnToList(iq_reconstruction_error)))
iq_reconstruction_error_test = iq_model.anomaly(test_data=iq_testing_frame, per_feature=False)
iq_reconstruction_error_test = list(map(float, h2OColumnToList(iq_reconstruction_error_test)))
iq_train['reconstruction_error'] = iq_reconstruction_error
iq_test['reconstruction_error'] = iq_reconstruction_error_test

print ""

sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
sj_testing_frame = h2o.H2OFrame(sj_test)
sj_testing_frame.set_names(list(sj_test.columns))
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))
iq_testing_frame = h2o.H2OFrame(iq_test)
iq_testing_frame.set_names(list(iq_test.columns))

print "SJ Training Model"
print "-----------------"
columns = list(sj_train.columns)
columns.remove('total_cases')
model_sj = H2OGradientBoostingEstimator(nfolds=5)
model_sj.train(x=columns, y='total_cases', training_frame=sj_training_frame)
sj_testing_frame = h2o.H2OFrame(sj_test)
sj_testing_frame.set_names(list(sj_test.columns))
prediction_sj = model_sj.predict(test_data=sj_testing_frame)

print "IQ Training Model"
print "-----------------"
columns = list(iq_train.columns)
columns.remove('total_cases')
model_iq = H2OGradientBoostingEstimator(nfolds=5)
model_iq.train(x=columns, y='total_cases', training_frame=iq_training_frame)
iq_testing_frame = h2o.H2OFrame(iq_test)
iq_testing_frame.set_names(list(iq_test.columns))
prediction_iq = model_iq.predict(test_data=iq_testing_frame)

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