import pandas as pd
import matplotlib.pyplot as plt
import h2o

correlation_list_shift = []
for shift in range(1,10):
    sj_train = pd.read_csv('sj_train_original.csv')
    iq_train = pd.read_csv('iq_train_original.csv')
    sj_test = pd.read_csv('sj_test_original.csv')
    iq_test = pd.read_csv('iq_train_original.csv')
    pd_submit = pd.read_csv('dataset/submission_format.csv')

    total_cases_sj = list(sj_train['total_cases'])
    total_cases_iq = list(iq_train['total_cases'])

    # Shift response column up
    total_cases_sj = total_cases_sj[shift:]
    total_cases_iq = total_cases_iq[shift:]

    # Adjust training data to match response column
    sj_train = sj_train[:-shift]
    iq_train = iq_train[:-shift]

    # Combine training columns with response column
    sj_train['total_cases'] = total_cases_sj
    iq_train['total_cases'] = total_cases_iq
    columns = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']

    correlation_list = []
    for column in columns:
        correlation = sj_train['total_cases'].corr(sj_train[column])
        correlation_list.append(correlation)
    correlation_list_shift.append(correlation_list)


for i in range(len(correlation_list_shift)):
    plt.plot(range(len(correlation_list_shift[i])), correlation_list_shift[i])
plt.legend(range(1, 20), loc='upper right')
plt.show()


