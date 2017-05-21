"""
Train separate models for two cities and save best two models
Missing values have filled using Gradient Boosting method
"""

import Dataset
import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator

# Configuration
n_iterations = 100

# Creating training frames
pd_train = pd.read_csv('na_filled_random_forest.csv')

sj_train = pd_train[pd_train['city'] == 'sj']
iq_train = pd_train[pd_train['city'] == 'iq']

del sj_train['city']
del iq_train['city']

sj_train.to_csv("sj_train.csv", index=False)
iq_train.to_csv("iq_train.csv", index=False)

# Define input output parameters
input_columns = Dataset.TRAINING_COLUMNS
response_column = Dataset.RESPONSE_COLUMN
input_columns.remove('city')

# Start h2o server
h2o.init(max_mem_size_GB=6)

# Create h2o frame
sj_training_frame = h2o.H2OFrame(sj_train)
sj_training_frame.set_names(list(sj_train.columns))
iq_training_frame = h2o.H2OFrame(iq_train)
iq_training_frame.set_names(list(iq_train.columns))

# Models
sj_min_mae = 1000
sj_best_model = None
iq_min_mae = 1000
iq_best_model = None

for i in range(n_iterations):
    model_sj = H2OGradientBoostingEstimator(nfolds=10)
    model_sj.train(x=input_columns, y=response_column, training_frame=sj_training_frame)

    model_iq = H2OGradientBoostingEstimator(nfolds=10)
    model_iq.train(x=input_columns, y=response_column, training_frame=iq_training_frame)

    if model_sj.mae() < sj_min_mae:
        sj_min_mae = model_sj.mae()
        sj_best_model = model_sj

    if model_iq.mae() < iq_min_mae:
        iq_min_mae = model_iq.mae()
        iq_best_model = model_iq

print("Model SJ MAE : " + str(sj_min_mae))
h2o.save_model(model=sj_best_model, path='sj_model', force=True)

print("Model IQ MAE : " + str(iq_min_mae))
h2o.save_model(model=iq_best_model, path='iq_model', force=True)








