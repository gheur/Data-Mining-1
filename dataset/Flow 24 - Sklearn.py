import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import GradientBoostingRegressor

pd_train = pd.read_csv('na_filled_random_forest.csv')

sj_train = pd_train[pd_train['city'] == 'sj']
iq_train = pd_train[pd_train['city'] == 'iq']

del sj_train['city']
del iq_train['city']

# Remove unnecessary features
del sj_train['ndvi_ne']
del iq_train['weekofyear']
del iq_train['ndvi_nw']

