import pandas as pd
from featureeng.parser import XMLParser
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator


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

# Load data sets into pandas framess
pd_training = pd.read_csv('na_filled.csv')
pd_labels = pd.read_csv('dataset/dengue_labels_train.csv')
pd_test = pd.read_csv('dataset/dengue_features_test.csv')
pd_submit = pd.read_csv('dataset/submission_format.csv')

pd_training['total_cases'] = pd_labels['total_cases']

fe_training = XMLParser.apply_feature_eng(pd_training, 'flow.xml')
fe_testing = XMLParser.apply_feature_eng(pd_test, 'flow.xml')

# Initialize H2O server
h2o.init(nthreads=-1,max_mem_size_GB=6)

# Create h2o frames
hd_train = h2o.H2OFrame(fe_training)
hd_train.set_names(list(fe_training.columns))
hd_test = h2o.H2OFrame(fe_testing)
hd_test.set_names(list(fe_testing.columns))

h2o.export_file(frame=hd_train, path='h2o_train_fe.csv', force=True)
h2o.export_file(frame=hd_test, path='h2o_test_fe.csv', force=True)

# Identifying columns
response_column = 'total_cases'
training_columns = list(fe_testing.columns)

# Defining machine learning model
# model = H2ODeepLearningEstimator(epochs=100, hidden=[200, 200], nfolds=10)
# model = H2ORandomForestEstimator(ntrees=100, max_depth=20, binomial_double_trees=True, nfolds=20)
model = H2ORandomForestEstimator(ntrees=100, max_depth=20, binomial_double_trees=True, nfolds=10)


# Train model
model.train(x=training_columns, y=response_column, training_frame=hd_train)
print(model.model_performance())

predictions = model.predict(test_data=hd_test)
predictions = h2OColumnToList(predictions, data_type='real')

print(type(predictions))
print(predictions)

for i in range(len(predictions)):
    if predictions[i] < 0.0:
        predictions[i] = 1
    else:
        predictions[i] = int(predictions[i] + 0.5)

pd_submit[response_column] = pd.Series(predictions)
pd_submit.to_csv("submit.csv", index=False)













