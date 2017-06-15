import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


def h2OColumnToList(data_column, data_type="real"):
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


def impute_missing_values(data_frame, columns):
    if not isinstance(data_frame, pd.DataFrame):
        return
    if not isinstance(columns, list):
        return

    # Impute summary
    impute_summary = {}

    # result frame
    result_frame = pd.DataFrame(data_frame)

    # Start h2o server
    h2o.init(max_mem_size_GB=5)
    for column in columns:
        print "Processing :", column

        # Defining columns
        response_column = column
        training_columns = list(data_frame.columns)
        training_columns.remove(response_column)

        # Creating h2o frame
        training_frame = h2o.H2OFrame(data_frame)
        training_frame.set_names(list(data_frame.columns))

        # Defining model
        model = H2ORandomForestEstimator(ntrees=75, max_depth=25, nbins=25, binomial_double_trees=True, nfolds=10)
        model.train(x=training_columns, y=response_column, training_frame=training_frame)

        # Predict values
        predictions = model.predict(test_data=training_frame)
        predictions = list(map(float, h2OColumnToList(predictions)))

        # Add predictions to the result frame
        result_frame[column] = predictions

        actual = data_frame[column]
        predicted = result_frame[column]

        rmse = sqrt(mean_squared_error(actual, predicted))
        impute_summary[column] = ('RMSE', rmse)

    # Removing all processes
    h2o.remove_all()

    # Displaying impute summary
    for key in impute_summary:
        print impute_summary[key], key

    return result_frame


#Loading data frames
sj_train = pd.read_csv('iq_train_original.csv')
iq_train = pd.read_csv('sj_train_original.csv')
sj_test = pd.read_csv('sj_test_original.csv')
iq_test = pd.read_csv('iq_test_original.csv')

# Setting up columns
response_column = 'total_cases'
input_columns = ['year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw',
           'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
           'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
           'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
           'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
           'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg',
           'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
           'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']

sj_columns = list(input_columns)
iq_columns = list(input_columns)

# Imputing missing values for SJ Training
sj_columns.remove('year')
sj_columns.remove('weekofyear')
sj_columns.remove('week_start_date')
sj_train = pd.DataFrame(impute_missing_values(sj_train, sj_columns))

# Imputing missing values for SJ Testing
sj_test = pd.DataFrame(impute_missing_values(sj_test, sj_columns))

# Imputing missing values for IQ Training
iq_columns.remove('year')
iq_columns.remove('weekofyear')
iq_columns.remove('week_start_date')
iq_train = pd.DataFrame(impute_missing_values(iq_train, iq_columns))

# Imputing missing values for IQ Testing
iq_test = pd.DataFrame('')


