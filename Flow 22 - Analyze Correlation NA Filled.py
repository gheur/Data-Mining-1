import pandas as pd

pd_train = pd.read_csv('na_filled_random_forest.csv')

sj_train = pd_train[pd_train['city'] == 'sj']
iq_train = pd_train[pd_train['city'] == 'iq']

del sj_train['city']
del iq_train['city']

features = list(pd_train.columns)
features.remove('total_cases')
features.remove('week_start_date')
features.remove('city')

print("SJ Train")
print("--------")
sj_feature_correlation_list =[]
for feature in features:
    correlation = sj_train['total_cases'].corr(sj_train[feature])
    sj_feature_correlation_list.append((feature, abs(correlation)))

sj_sorted_list = sorted(sj_feature_correlation_list, key=lambda tup: tup[1])
sj_sorted_list = sj_sorted_list[::-1]

for item in sj_sorted_list:
    print(item)

print("IQ Train")
print("--------")

iq_feature_correlation_list =[]
for feature in features:
    correlation = iq_train['total_cases'].corr(iq_train[feature])
    iq_feature_correlation_list.append((feature, abs(correlation)))

iq_sorted_list = sorted(iq_feature_correlation_list, key=lambda tup: tup[1])
iq_sorted_list = iq_sorted_list[::-1]

for item in iq_sorted_list:
    print(item)
