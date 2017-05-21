
# Simple prediction
import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator
from featureeng.parser import XMLParser


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
        return map(float, splitter_list)
    elif data_type == "enum":
        return splitter_list


def NormalizeScale(data_frame, column_names):
    """
    Normalize selected columns
    :param data_frame: input data frame         [pandas.DataFrame]
    :param column_names: selected column names  [list]
    :return: normalized frame                   [pandas.DataFrame]
    """

    for column in column_names:
        data_frame[column] = (data_frame[column] - data_frame[column].mean()) / data_frame[column].std()

    return data_frame


def MinMaxScale(data_frame, column_names):
    """
    Min Max scale selected columns
    :param data_frame: input data frame                     [pandas.DataFrame]
    :param min_value: minimum value of the given series     [int]
    :param max_value: maximum value of the given series     [int]
    :return: min max scaled frame
    """

    for column in column_names:
        data_frame[column] = (data_frame[column] - data_frame[column].min()) / \
                             (data_frame[column].max() - data_frame[column].min())

    return data_frame


def ResolveTestDataTypeCompatibility(hd_train, hd_test):
    """

    :param hd_train:
    :param hd_test:
    :return:
    """
    column_names = list(hd_test.names)
    for column in column_names:
        d_type = str(hd_train.type(col=column))
        if str(hd_test.type(col=column)) != d_type:
            if d_type == 'enum':
                hd_test[column] = hd_test[column].asfactor()
            elif d_type == 'real':
                hd_test[column] = hd_test[column].asnumeric()

    return hd_test

# Initialize h2o server
h2o.init(nthreads=-1)

# Load data sets
pd_train = pd.read_csv('na_filled.csv')
pd_labels = pd.read_csv('dataset/dengue_labels_train.csv')
pd_test = pd.read_csv('dataset/dengue_features_test.csv')
pd_submit = pd.read_csv('dataset/submission_format.csv')

norm_columns = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c',
                'station_precip_mm']

pd_train = NormalizeScale(pd_train, norm_columns)
pd_test = NormalizeScale(pd_test, norm_columns)

# pd_train = XMLParser.apply_feature_eng(pd_train, 'flow.xml')
# pd_test = XMLParser.apply_feature_eng(pd_test, 'flow.xml')

# Identifying columns
response_column = 'total_cases'
training_columns = list(pd_test.columns)

# Merging labels with training data
pd_train[response_column] = pd_labels[response_column]

# Create h2o frames
hd_train = h2o.H2OFrame(pd_train)
hd_train.set_names(list(pd_train.columns))
hd_test = h2o.H2OFrame(pd_test)
hd_test.set_names(list(pd_test.columns))

# Check type compatibility between train test data
hd_test = ResolveTestDataTypeCompatibility(hd_train, hd_test)

h2o.export_file(frame=hd_train, path='h2o_train.csv', force=True)
h2o.export_file(frame=hd_test, path='h2o_test.csv', force=True)

# Defining machine learning model
# model = H2ORandomForestEstimator(ntrees=100, max_depth=20, binomial_double_trees=True, nfolds=20)
# model = H2ORandomForestEstimator(ntrees=100, max_depth=20, binomial_double_trees=True)
model = H2OGradientBoostingEstimator()

# Train model
model.train(x=training_columns, y=response_column, training_frame=hd_train)
print model.model_performance()

predictions = model.predict(test_data=hd_test)
predictions = h2OColumnToList(predictions, data_type='real')

for i in range(len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = 1
    else:
        predictions[i] = int(predictions[i] + 0.5)

pd_submit[response_column] = pd.Series(predictions)
pd_submit.to_csv("submit.csv", index=False)











