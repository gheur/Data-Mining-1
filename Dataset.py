import pandas as pd

# Paths
TRAINING_LABELS_PATH = 'dataset/dengue_labels_train.csv'
TRAINING_FRAME_PATH = 'dataset/dengue_features_train.csv'
TESTING_FRAME_PATH = 'dataset/dengue_features_test.csv'
SUBMIT_FRAME_PATH = 'dataset/submission_format.csv'
NA_FILLED_FRAME_PATH = 'na_filled.csv'

# Data frames
PD_TRAIN = pd.read_csv(TRAINING_FRAME_PATH)
PD_LABEL = pd.read_csv(TRAINING_LABELS_PATH)
PD_TEST = pd.read_csv(TESTING_FRAME_PATH)
PD_SUBMIT = pd.read_csv(SUBMIT_FRAME_PATH)
PD_NA_FILLED = pd.read_csv(NA_FILLED_FRAME_PATH)

# Features
RESPONSE_COLUMN = 'total_cases'
TRAINING_COLUMNS = list(PD_TRAIN.columns)

PD_TRAIN[RESPONSE_COLUMN] = PD_LABEL[RESPONSE_COLUMN]





