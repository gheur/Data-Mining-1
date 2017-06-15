# Simple prediction
import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2ORandomForestEstimator


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


# Initialize h2o server
h2o.init(nthreads=-1)

# Load data sets
pd_train = pd.read_csv('dataset/dengue_features_train.csv')
pd_labels = pd.read_csv('dataset/dengue_labels_train.csv')
pd_test = pd.read_csv('dataset/dengue_features_test.csv')
pd_submit = pd.read_csv('dataset/submission_format.csv')

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

h2o.export_file(frame=hd_train, path='h2o_train.csv', force=True)
h2o.export_file(frame=hd_test, path='h2o_test.csv', force=True)

# Defining machine learning model
# model = H2ODeepLearningEstimator(epochs=100, hidden=[200, 200], nfolds=10)
model = H2ORandomForestEstimator()

# Train model
model.train(x=training_columns, y=response_column, training_frame=hd_train)
print(model.model_performance())

predictions = model.predict(test_data=hd_test)
predictions = h2OColumnToList(predictions, data_type='real')

for i in range(len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = 1
    else:
        predictions[i] = int(predictions[i] + 0.5)

pd_submit[response_column] = pd.Series(predictions)
pd_submit.to_csv("submit.csv")











