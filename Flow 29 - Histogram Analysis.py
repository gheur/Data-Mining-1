import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv('fe_train_imputed.csv')
analyzing_columns = list(train.columns)
analyzing_columns.remove('ndvi_ne')
analyzing_columns.remove('ndvi_nw')
analyzing_columns.remove('ndvi_se')
analyzing_columns.remove('ndvi_sw')
analyzing_columns.remove('city')
analyzing_columns.remove('year')
analyzing_columns.remove('week_start_date')
analyzing_columns.remove('total_cases')

for column in analyzing_columns:
    series = np.array(train[column])
    plt.plot(series)
    plt.show()