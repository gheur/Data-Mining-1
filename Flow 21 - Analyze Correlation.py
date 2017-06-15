import pandas as pd
import matplotlib.pyplot as plt

sj_train = pd.read_csv('sj_train_original.csv')
iq_train = pd.read_csv('iq_train_original.csv')

features = list(sj_train.columns)
features.remove('total_cases')
features.remove('week_start_date')

# sj_feature_correlation_list =[]
# for feature in features:
#     correlation = sj_train['total_cases'].corr(sj_train[feature])
#     sj_feature_correlation_list.append((feature, abs(correlation)))
#
# sj_sorted_list = sorted(sj_feature_correlation_list, key=lambda tup: tup[1])
# sj_sorted_list = sj_sorted_list[::-1]
#
# for item in sj_sorted_list:
#     print(item)

iq_feature_correlation_list =[]
for feature in features:
    correlation = iq_train['total_cases'].corr(iq_train[feature])
    iq_feature_correlation_list.append((feature, abs(correlation)))

iq_sorted_list = sorted(iq_feature_correlation_list, key=lambda tup: tup[1])
iq_sorted_list = iq_sorted_list[::-1]

for item in iq_sorted_list:
    print(item)

'''
SJ Train
('weekofyear', 0.28713422267094074)
('year', 0.21268978104190459)
('reanalysis_specific_humidity_g_per_kg', 0.20794740781198034)
('reanalysis_dew_point_temp_k', 0.2037742415881591)
('station_avg_temp_c', 0.19661656049148882)
('reanalysis_max_air_temp_k', 0.19453181561745561)
('station_max_temp_c', 0.18990073139980398)
('reanalysis_min_air_temp_k', 0.18794289083152466)
('reanalysis_air_temp_k', 0.18191694846733036)
('station_min_temp_c', 0.17701193575977914)
('reanalysis_avg_temp_k', 0.17526745180821787)
('reanalysis_relative_humidity_percent', 0.14404469835447362)
('reanalysis_precip_amt_kg_per_m2', 0.10745737870210094)
('ndvi_nw', 0.07530714420074977)
('reanalysis_tdtr_k', 0.067599928952445043)
('reanalysis_sat_precip_amt_mm', 0.060210670714737284)
('precipitation_amt_mm', 0.060210670714737284)
('station_precip_mm', 0.051759212768865762)
('ndvi_ne', 0.037639462842305155)
('station_diur_temp_rng_c', 0.034630067928700235)
('ndvi_se', 0.0011127769818490402)
('ndvi_sw', 0.00033319221111510088)
'''

'''
IQ Train
('reanalysis_specific_humidity_g_per_kg', 0.236476051496291)
('reanalysis_dew_point_temp_k', 0.23040141314797838)
('reanalysis_min_air_temp_k', 0.21451395224156697)
('station_min_temp_c', 0.21170235486678332)
('year', 0.17945112139969918)
('reanalysis_tdtr_k', 0.1344245275619895)
('reanalysis_relative_humidity_percent', 0.13008273697398187)
('station_avg_temp_c', 0.11306976857499454)
('reanalysis_precip_amt_kg_per_m2', 0.10117149447898517)
('reanalysis_air_temp_k', 0.097097788415201655)
('reanalysis_sat_precip_amt_mm', 0.090170509404735796)
('precipitation_amt_mm', 0.090170509404735796)
('reanalysis_avg_temp_k', 0.079871736901985568)
('station_max_temp_c', 0.075278508684555295)
('station_diur_temp_rng_c', 0.058229712800654046)
('reanalysis_max_air_temp_k', 0.056473574604560456)
('station_precip_mm', 0.042975521369309294)
('ndvi_se', 0.041066599868674988)
('ndvi_sw', 0.032999182684033367)
('ndvi_ne', 0.020215071033071963)
('weekofyear', 0.01185049160167872)
('ndvi_nw', 0.0095857465737125246)
'''