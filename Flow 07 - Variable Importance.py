import h2o
import Dataset
from h2o.estimators import H2ORandomForestEstimator
from tabulate import tabulate


def display(variable_map):
    if not isinstance(variable_map, dict):
        return

    variable_map = sorted(variable_map.items(), key=lambda x: x[1])[::-1]
    print(tabulate(variable_map, headers=["Name", "Value"]))
    # for key in variable_map:
    #     print(key)


def sameModel():
    # Configurations
    n_iterations = 10

    # Initialize server
    h2o.init(nthreads=-1)

    # Load data frames
    h_train = h2o.H2OFrame(Dataset.PD_TRAIN)

    # Initialize variable map
    variable_map = {}
    for key in Dataset.TRAINING_COLUMNS:
        variable_map[key] = 0.0

    # Same model for several iterations
    for i in range(n_iterations):
        # defining features
        model = H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True)
        model.train(x=Dataset.TRAINING_COLUMNS, y=Dataset.RESPONSE_COLUMN, training_frame=h_train)

        variable_importance = model.varimp()
        for variable in variable_importance:
            variable_map[str(variable[0])] += variable[2]

    display(variable_map)


def differentModels():
    # Initialize server
    h2o.init(nthreads=-1)

    # Load data frames
    h_train = h2o.H2OFrame(Dataset.PD_TRAIN)

    # Initialize variable map
    variable_map = {}
    for key in Dataset.TRAINING_COLUMNS:
        variable_map[key] = 0.0

    models = []
    models.append(H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True))
    models.append(H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True, ntrees=20))
    models.append(H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True, ntrees=30))
    models.append(H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True, ntrees=40))
    models.append(H2ORandomForestEstimator(nfolds=10, binomial_double_trees=True, ntrees=50))

    # Unique model for each iteration
    for model in models:
        # defining features

        model.train(x=Dataset.TRAINING_COLUMNS, y=Dataset.RESPONSE_COLUMN, training_frame=h_train)

        variable_importance = model.varimp()
        for variable in variable_importance:
            variable_map[str(variable[0])] += variable[2]

    display(variable_map)

sameModel()

