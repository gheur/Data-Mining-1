# Analyze data using featuring.presenting Chart : Progressing

from featureeng.presenting import Chart
import pandas as pd

# Load data sets
pd_train = pd.read_csv('dataset/dengue_features_train.csv')
pd_labels = pd.read_csv('dataset/dengue_labels_train.csv')

print( min(pd_labels['total_cases']))

# # Merging labels with training data
# pd_train['total_cases'] = pd_labels['total_cases']
# cols = ['year', 'weekofyear', 'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm', 'total_cases']
#
# for col in cols:
#     name = col + '.png'
#     print "Generating", name
#     Chart.saveChart(data_frame=pd_train, columns=[col, 'total_cases'], file_name=name, scaling='minmax')
