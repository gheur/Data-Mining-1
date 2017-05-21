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

# Define input output parameters
input_columns = Dataset.TRAINING_COLUMNS
response_column = Dataset.RESPONSE_COLUMN
input_columns.remove('city')

# Start h2o server
h2o.init()

# Create h2o frame
training_frame = h2o.H2OFrame(pd_train)
training_frame.set_names(list(pd_train.columns))

# Measurements
mae = [] # Mean Absolute Errors for model
rmse = [] # Root Mean Squared Errors for model

for i in range(n_iterations):
    model = H2OGeneralizedLinearEstimator(nfolds=10)
    model.train(x=input_columns, y=response_column, training_frame=training_frame)

    mae.append(model.mae())
    rmse.append(model.rmse())

print("Model : Single")
print("--------------")
print("Average MAE       : " + str(numpy.average(mae)))
print("Average RMSE      : " + str(numpy.average(rmse)))
print("MAE Standard Dev  : " + str(numpy.std(mae)))
print("RMSE Standard Dev : " + str(numpy.std(rmse)))









