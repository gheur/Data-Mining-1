"""
Train separate models for two cities assuming their behaviors are different
When predicting the results, model will be selected at the runtime based on the city
"""

import Dataset
import h2o
import numpy
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator, \
    H2ODeepLearningEstimator, H2OGeneralizedLinearEstimator

# Configuration
n_iterations = 10

# Creating training frames
pd_train = Dataset.PD_TRAIN

sj_train = pd_train[pd_train['city'] == 'sj']
iq_train = pd_train[pd_train['city'] == 'iq']

del sj_train['city']
del iq_train['city']

# Define input output parameters
input_columns = Dataset.TRAINING_COLUMNS
response_column = Dataset.RESPONSE_COLUMN
input_columns.remove('city')

# Start h2o server
h2o.init()

# Create h2o frame
sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))

# Measurements
sj_mae = [] # Mean Absolute Errors for sj model
iq_mae = [] # Mean Absolute Errors for iq model
sj_rmse = [] # Root Mean Squared Errors for sj model
iq_rmse = [] # Root Mean Squared Errors for iq model

for i in range(n_iterations):
    model_sj = H2ODeepLearningEstimator(nfolds=10, hidden=[32, 32, 32, 32, 32, 32, 32, 32])
    model_sj.train(x=input_columns, y=response_column, training_frame=sj_training_frame)

    model_iq = H2ODeepLearningEstimator(nfolds=10, hidden=[32, 32, 32, 32, 32, 32, 32, 32])
    model_iq.train(x=input_columns, y=response_column, training_frame=iq_training_frame)

    sj_mae.append(model_sj.mae())
    sj_rmse.append(model_sj.rmse())
    iq_mae.append(model_iq.mae())
    iq_rmse.append(model_iq.rmse())

print("Model : SJ")
print("----------")
print("Average MAE       : " + str(numpy.average(sj_mae)))
print("Average RMSE      : " + str(numpy.average(sj_rmse)))
print("MAE Standard Dev  : " + str(numpy.std(sj_mae)))
print("RMSE Standard Dev : " + str(numpy.std(sj_rmse)))
print()

print("Model : IQ")
print("----------")
print("Average MAE       : " + str(numpy.average(iq_mae)))
print("Average RMSE      : " + str(numpy.average(iq_rmse)))
print("MAE Standard Dev  : " + str(numpy.std(iq_mae)))
print("RMSE Standard Dev : " + str(numpy.std(iq_rmse)))








