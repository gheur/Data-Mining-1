import pandas as pd
import Dataset
import h2o
import Dataset

from h2o.estimators import H2OGradientBoostingEstimator, H2ODeepLearningEstimator, H2ORandomForestEstimator, \
    H2OGeneralizedLinearEstimator

n_iterations = 10
max_shift = 10

sj_train = pd.read_csv('sj_train_original.csv')
iq_train = pd.read_csv('iq_train_original.csv')

total_cases_sj = list(sj_train['total_cases'])
total_cases_iq = list(iq_train['total_cases'])

for shift in range(1, max_shift+1):
    # Make connection to h2o server
    h2o.init(nthreads=-1, max_mem_size_GB=8)
    h2o.remove_all()

    # Shift response column up
    total_cases_sj = total_cases_sj[shift:]
    total_cases_iq = total_cases_iq[shift:]

    # Adjust training data to match response column
    sj_train = sj_train[:-shift]
    iq_train = iq_train[:-shift]

    # Combine training columns with response column
    sj_train['total_cases'] = total_cases_sj
    iq_train['total_cases'] = total_cases_iq

    # Create h2o frames
    sj_training_frame = h2o.H2OFrame(sj_train)
    sj_training_frame.set_names(list(sj_train.columns))
    iq_training_frame = h2o.H2OFrame(iq_train)
    iq_training_frame.set_names(list(iq_train.columns))

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
        model_sj = H2OGradientBoostingEstimator(nfolds=5)
        model_sj.train(x=input_columns, y=response_column, training_frame=sj_training_frame)

        model_iq = H2OGradientBoostingEstimator(nfolds=5)
        model_iq.train(x=input_columns, y=response_column, training_frame=iq_training_frame)

        if model_sj.mae() < sj_min_mae:
            sj_min_mae = model_sj.mae()
            sj_best_model = model_sj

        if model_iq.mae() < iq_min_mae:
            iq_min_mae = model_iq.mae()
            iq_best_model = model_iq

    print("Shift :", shift)

    print("Model SJ MAE  : " + str(sj_min_mae))
    h2o.save_model(model=sj_best_model, path='sj_model', force=True)

    print("Model IQ MAE  : " + str(iq_min_mae))
    h2o.save_model(model=iq_best_model, path='iq_model', force=True)










