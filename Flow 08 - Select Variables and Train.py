import pandas as pd
import h2o
import Dataset
from h2o.estimators import H2ORandomForestEstimator


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


selected_variables = ['ndvi_se', 'ndvi_sw', 'week_start_date', 'year', 'weekofyear', 'ndvi_nw',
                      'reanalysis_min_air_temp_k', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_air_temp_k',
                      'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_tdtr_k']

pd_train = Dataset.PD_NA_FILLED
pd_test = Dataset.PD_TEST
pd_label = Dataset.PD_LABEL
pd_submit = Dataset.PD_SUBMIT

# Initialize new frame
pd_selected = pd.DataFrame()

# Copy relevant columns from the original data set
for column in selected_variables:
    pd_selected[column] = pd_train[column]

pd_selected[Dataset.RESPONSE_COLUMN] = pd_label[Dataset.RESPONSE_COLUMN]

# Defining features
training_columns = list(pd_selected.columns)
response_column = Dataset.RESPONSE_COLUMN

# Start h2o server
h2o.init(nthreads=-1)

# Create h2o frames
h_train = h2o.H2OFrame(pd_selected)
h_train.set_names(list(pd_selected.columns))

h_test = h2o.H2OFrame(pd_test)
h_test.set_names(list(pd_test.columns))

# Define h2o model
model = H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True)
model.train(x=training_columns, y=response_column, training_frame=h_train)
predictions = model.predict(test_data=h_test)
predictions = h2OColumnToList(predictions, data_type='real')

for i in range(len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = 1
    else:
        predictions[i] = int(predictions[i] + 0.5)

pd_submit[response_column] = pd.Series(predictions)
pd_submit.to_csv("submit.csv", index=False)

